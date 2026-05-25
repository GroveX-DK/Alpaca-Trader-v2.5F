"""Hent OHLCV-historik fra Alpaca med fejlhåndtering og valgfri disk-cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from stock_predictor import config
from stock_predictor.feature_engineer import rolling_annualized_log_vol_pct
from stock_predictor.watchlist_metrics import compute_watchlist_metrics

logger = logging.getLogger(__name__)


def _ts(d: date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.upper()}.parquet"


def required_last_bar_date(as_of: date | None = None) -> date:
    """
    Sidste forventede fulde daglige bar (US-aktier).

    Weekender → seneste fredag; hverdage → forrige handelsdag (BDay),
    så fredag morgen ikke kræver tail-API når cache slutter torsdag.
    """
    ts = pd.Timestamp(as_of or date.today()).normalize()
    if ts.weekday() >= 5:
        ts = ts - pd.offsets.BDay(1)
    else:
        ts = ts - pd.offsets.BDay(1)
    return ts.date()


def _trim_cached_window(
    cached: pd.DataFrame,
    want_start: date,
    want_end: date,
) -> pd.DataFrame:
    if cached.empty:
        return cached
    want_start_ts = _ts(want_start)
    want_end_ts = _ts(want_end)
    mask = (cached.index >= want_start_ts) & (cached.index <= want_end_ts)
    return cached.loc[mask].sort_index()


def _cache_covers_window(
    cached: pd.DataFrame,
    want_start: date,
    want_end: date,
    required_end: date,
) -> bool:
    """True hvis cache dækker [want_start, want_end] og har bar senest required_end."""
    if cached is None or cached.empty:
        return False
    required_ts = _ts(required_end)
    if _ts(cached.index.max()) < required_ts:
        return False
    if _ts(cached.index.min()) > _ts(want_start):
        return False
    trimmed = _trim_cached_window(cached, want_start, want_end)
    return not trimmed.empty


@dataclass(frozen=True)
class FetchBarsResult:
    bars: Dict[str, pd.DataFrame]
    cache_only: bool
    required_end: date


def _with_vol_annual_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Tilføj/opdatér annualiseret vol fra daglige log-afkast (Alpaca: lukkekurs)."""
    if df.empty:
        return df
    out = df.copy()
    out["vol_annual_pct"] = rolling_annualized_log_vol_pct(out["close"])
    return out


def _enrich_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Watchlist-metriker + vol_annual_pct (fuld serie, fx før gem/trim)."""
    if df.empty:
        return df
    return _with_vol_annual_pct(compute_watchlist_metrics(df))


def _read_cache_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)
        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df = df.sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return None
        cols = ["open", "high", "low", "close", "volume"]
        for opt in ("vol_annual_pct", "vix_close"):
            if opt in df.columns:
                cols.append(opt)
        out = df[cols].copy()
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = out[c].astype(float)
        for opt in ("vol_annual_pct", "vix_close"):
            if opt in out.columns:
                out[opt] = out[opt].astype(float)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke læse cache %s: %s", path, exc)
        return None


def _atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=True)
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def _bars_list_to_df(sym_bars) -> pd.DataFrame:
    rows = []
    for bar in sym_bars:
        rows.append(
            {
                "date": pd.Timestamp(bar.timestamp).normalize().tz_localize(None),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")
    return df.set_index("date").sort_index()


def _fetch_symbol_range(
    client: StockHistoricalDataClient,
    symbol: str,
    start: date,
    end: date,
) -> Optional[pd.DataFrame]:
    if start > end:
        return pd.DataFrame()
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
        )
        bars = client.get_stock_bars(req)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alpaca get_stock_bars fejlede for %s [%s..%s]: %s", symbol, start, end, exc)
        return None

    if symbol not in bars.data or not bars.data[symbol]:
        return pd.DataFrame()
    return _bars_list_to_df(bars.data[symbol])


def _fetch_all_symbols_batch(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date,
    end: date,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not symbols or start > end:
        return out
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
        )
        bars = client.get_stock_bars(req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Alpaca get_stock_bars fejlede (batch): %s", exc)
        return out

    for sym in symbols:
        try:
            if sym not in bars.data or not bars.data[sym]:
                logger.warning("Ingen datapunkter for %s.", sym)
                continue
            df = _bars_list_to_df(bars.data[sym])
            if df.empty:
                continue
            out[sym] = _enrich_ohlcv(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kunne ikke konvertere data for %s: %s", sym, exc)
    return out


def _merge_trim_save_symbol(
    symbol: str,
    want_start: date,
    want_end: date,
    client: StockHistoricalDataClient,
    cache_dir: Path,
    *,
    required_end: date | None = None,
) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, symbol)
    cached = _read_cache_parquet(path)

    want_start_ts = _ts(want_start)
    want_end_ts = _ts(want_end)
    req_end = required_end if required_end is not None else required_last_bar_date(want_end)
    req_end_ts = _ts(req_end)

    def trim(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = (df.index >= want_start_ts) & (df.index <= want_end_ts)
        return df.loc[mask].sort_index()

    if cached is None or cached.empty:
        fresh = _fetch_symbol_range(client, symbol, want_start, want_end)
        if fresh is None:
            return None
        full = _enrich_ohlcv(fresh)
        merged = trim(full)
        if not merged.empty:
            try:
                _atomic_save_parquet(full, path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Kunne ikke skrive cache for %s: %s", symbol, exc)
        return merged if not merged.empty else None

    cmin = cached.index.min()
    cmax = cached.index.max()
    parts: list[pd.DataFrame] = []

    if _ts(cmin) > want_start_ts:
        back_end = (_ts(cmin) - pd.Timedelta(days=1)).date()
        left = _fetch_symbol_range(client, symbol, want_start, back_end)
        if left is None:
            logger.warning("Backfill-api fejlede for %s — bruger kun cache hvor muligt.", symbol)
        elif not left.empty:
            parts.append(left)

    parts.append(cached)

    if _ts(cmax) < req_end_ts:
        tail_start = (_ts(cmax) + pd.Timedelta(days=1)).date()
        right = _fetch_symbol_range(client, symbol, tail_start, req_end)
        if right is None:
            logger.warning("Tail-api fejlede for %s — bruger cache til sidste kendte bar.", symbol)
        elif not right.empty:
            parts.append(right)

    merged = pd.concat(parts, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    merged_full = _enrich_ohlcv(merged)

    try:
        _atomic_save_parquet(merged_full, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke opdatere cache for %s: %s", symbol, exc)

    out = trim(merged_full)
    return out if not out.empty else None


def fetch_daily_bars(
    api_key: str,
    secret_key: str,
    symbols: list[str],
    end: Optional[date] = None,
    lookback_calendar_days: int = 365,
    extra_buffer_days: int = 60,
    *,
    cache_enabled: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
    prefer_cache_only: bool = True,
) -> FetchBarsResult:
    """
    Returner OHLCV pr. symbol (med watchlist-metriker + vol_annual_pct).

    Cache-first: hvis Parquet dækker vinduet gennem required_last_bar_date(end),
    bruges kun disk-cache (ingen Alpaca-klient). API kaldes kun for symboler med
    forældet/manglende cache, når nøgler findes.

    Ved cache_enabled=False hentes hele [start, end] i ét batch-kald (kræver nøgler).
    """
    if end is None:
        end = date.today()

    start = end - timedelta(days=int(lookback_calendar_days + extra_buffer_days))
    required_end = required_last_bar_date(end)
    dfs: Dict[str, pd.DataFrame] = {}

    use_cache = config.OHLCV_CACHE_ENABLED if cache_enabled is None else cache_enabled
    cdir = config.CACHE_DIR if cache_dir is None else Path(cache_dir)

    symbols = sorted({s.strip().upper() for s in symbols if s.strip()})

    if not symbols:
        return FetchBarsResult(bars=dfs, cache_only=True, required_end=required_end)

    has_keys = bool(api_key and secret_key)

    if not use_cache:
        if not has_keys:
            logger.error("Manglende ALPACA_API_KEY eller ALPACA_SECRET_KEY (cache slået fra).")
            return FetchBarsResult(bars=dfs, cache_only=False, required_end=required_end)
        try:
            client = StockHistoricalDataClient(api_key, secret_key)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Kunne ikke oprette Alpaca-dataklient: %s", exc)
            return FetchBarsResult(bars=dfs, cache_only=False, required_end=required_end)
        return FetchBarsResult(
            bars=_fetch_all_symbols_batch(client, symbols, start, end),
            cache_only=False,
            required_end=required_end,
        )

    needs_api: List[str] = []

    for sym in symbols:
        cached = _read_cache_parquet(_cache_path(cdir, sym))
        if cached is not None and _cache_covers_window(cached, start, end, required_end):
            trimmed = _trim_cached_window(cached, start, end)
            dfs[sym] = _enrich_ohlcv(trimmed)
            logger.debug("Cache tilstrækkelig for %s (max=%s, krævet>=%s).", sym, cached.index.max().date(), required_end)
            continue
        needs_api.append(sym)

    if not needs_api:
        logger.info(
            "OHLCV kun fra disk-cache (%s symboler; sidste forventede bar %s). Ingen Alpaca-data-API.",
            len(dfs),
            required_end,
        )
        return FetchBarsResult(bars=dfs, cache_only=True, required_end=required_end)

    if not has_keys or prefer_cache_only:
        stale: List[str] = []
        for sym in needs_api:
            fallback = _read_cache_parquet(_cache_path(cdir, sym))
            if fallback is not None and not fallback.empty:
                trimmed = _trim_cached_window(fallback, start, end)
                if not trimmed.empty:
                    dfs[sym] = _enrich_ohlcv(trimmed)
                    logger.warning(
                        "Bruger forældet cache for %s (max=%s, krævet>=%s%s).",
                        sym,
                        fallback.index.max().date(),
                        required_end,
                        "; ingen API-nøgler" if not has_keys else "; prefer_cache_only",
                    )
                    continue
            stale.append(sym)
        if stale:
            logger.error(
                "Manglende/frisk cache for %s symboler%s: %s",
                len(stale),
                " uden API-nøgler" if not has_keys else " (prefer_cache_only)",
                ", ".join(stale),
            )
        cache_only_run = not needs_api or (prefer_cache_only and not stale)
        if prefer_cache_only and dfs:
            logger.info(
                "OHLCV fra disk-cache (%s symboler; %s uden frisk tail%s).",
                len(dfs),
                len(needs_api),
                "" if not stale else f", {len(stale)} udeladt",
            )
        return FetchBarsResult(
            bars=dfs,
            cache_only=cache_only_run and bool(dfs),
            required_end=required_end,
        )

    try:
        client = StockHistoricalDataClient(api_key, secret_key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Kunne ikke oprette Alpaca-dataklient: %s", exc)
        for sym in needs_api:
            fallback = _read_cache_parquet(_cache_path(cdir, sym))
            if fallback is not None and not fallback.empty:
                trimmed = _trim_cached_window(fallback, start, end)
                if not trimmed.empty:
                    dfs[sym] = _enrich_ohlcv(trimmed)
                    logger.warning("Bruger kun cache for %s (klient fejlede).", sym)
        return FetchBarsResult(bars=dfs, cache_only=False, required_end=required_end)

    logger.info(
        "Opdaterer %s symboler via Alpaca (cache frisk for %s; sidste forventede bar %s).",
        len(needs_api),
        len(dfs),
        required_end,
    )

    for sym in needs_api:
        try:
            df = _merge_trim_save_symbol(
                sym, start, end, client, cdir, required_end=required_end
            )
            if df is not None and not df.empty:
                dfs[sym] = df
                continue
            fallback = _read_cache_parquet(_cache_path(cdir, sym))
            if fallback is not None and not fallback.empty:
                trimmed = _trim_cached_window(fallback, start, end)
                if not trimmed.empty:
                    dfs[sym] = _enrich_ohlcv(trimmed)
                    logger.warning("Bruger kun cache for %s (api/cache-merge utilstrækkelig).", sym)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache/API-sti fejlede for %s: %s", sym, exc)
            fb = _read_cache_parquet(_cache_path(cdir, sym))
            if fb is not None and not fb.empty:
                trimmed = _trim_cached_window(fb, start, end)
                if not trimmed.empty:
                    dfs[sym] = _enrich_ohlcv(trimmed)

    return FetchBarsResult(bars=dfs, cache_only=False, required_end=required_end)
