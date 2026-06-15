"""Markeds-brede krise-signal-features (samme for alle symboler, gemt som vix_close).

Bygger én markeds-bred frame (date -> kolonner i config.MACRO_FEATURE_COLUMNS):

  vix_ts_slope     = ^VIX / ^VIX3M            (>1 = backwardation/panik)
  vvix_level       = ^VVIX / 100              (vol-of-vol-niveau)
  breadth_pct      = andel af watchlist over 200d MA          (fra cachen)
  xsec_corr        = middel parvis korrelation af 21d-afkast  (fra cachen)
  credit_ratio_chg = 5d pct-ændring i HYG/LQD (negativ i kredit-stress)
  move_chg         = 5d pct-ændring i ^MOVE   (positiv i bond-vol-stress)

VIX-familien, ^MOVE og HYG/LQD hentes via yfinance (samme kilde som ^VIX i forvejen).
Breadth + tværsnits-korrelation beregnes fra de eksisterende Parquet-cache-lukkekurser
(ingen ekstern kilde). Kolonner hvis kilde fejler udelades — feature-laget fylder dem
neutralt (se config.MACRO_FEATURE_NEUTRAL), så rækker aldrig droppes.

Frame caches til config.MACRO_CACHE_PATH og bæres ind i hver tickers OHLCV-cache via
build_dataset_frame (se tools/backfill_macro_features.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor import config
from stock_predictor.data_fetcher import _atomic_save_parquet, _cache_path
from stock_predictor.feature_engineer import rolling_annualized_log_vol_pct

logger = logging.getLogger(__name__)


def _to_naive_daily(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx)
    if getattr(out, "tz", None) is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _yf_close(ticker: str, period: str = "max") -> pd.Series | None:
    """Hent daglig close for ét yfinance-symbol; None hvis utilgængelig/tom."""
    try:
        import yfinance as yf

        raw = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True,
            progress=False, threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance-download fejlede for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        logger.warning("Ingen data for %s fra yfinance.", ticker)
        return None
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        # group_by default -> ('Close', ticker) eller (ticker, 'Close').
        lvl0 = {str(c).lower() for c in cols.get_level_values(0)}
        if "close" in lvl0:
            sub = raw.xs("Close", axis=1, level=0)
        else:
            sub = raw.xs("Close", axis=1, level=1)
        s = sub.iloc[:, 0]
    else:
        lower = {str(c).lower(): c for c in cols}
        if "close" not in lower:
            return None
        s = raw[lower["close"]]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    s.index = _to_naive_daily(s.index)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _watchlist_close_frame(cache_dir: Path) -> pd.DataFrame:
    """Bred close-frame (date x symbol) samlet fra Parquet-cachen til breadth/korrelation."""
    closes: dict[str, pd.Series] = {}
    for sym in config.WATCHLIST:
        path = _cache_path(cache_dir, sym)
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path, columns=["close"])
        except Exception:  # noqa: BLE001 - kolonne-only kan fejle for gamle filer
            try:
                df = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Springer %s over (cache-læsning): %s", sym, exc)
                continue
            if "close" not in df.columns:
                continue
            df = df[["close"]]
        if df.empty:
            continue
        df.index = _to_naive_daily(df.index)
        closes[sym] = pd.to_numeric(df["close"], errors="coerce")
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index()


def _breadth(closes: pd.DataFrame, ma_days: int) -> pd.Series:
    """Andel af symboler med close over deres egne ``ma_days``-glidende gennemsnit (0..1)."""
    ma = closes.rolling(window=ma_days, min_periods=ma_days).mean()
    above = closes > ma
    valid = ma.notna()
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (above & valid).sum(axis=1) / denom


def _xsec_corr(closes: pd.DataFrame, window: int) -> pd.Series:
    """Middel parvis korrelation (implied-correlation-estimator, ligevægt) over et vindue.

    avg_corr = (σ_p² − Σwᵢ²σᵢ²) / (Σᵢ≠ⱼ wᵢwⱼσᵢσⱼ), wᵢ = 1/N. Billigt (ingen N×N-matrix
    pr. dag): bruger rullende std pr. symbol + rullende std af ligevægts-indeksafkast.
    """
    rets = np.log(closes / closes.shift(1))
    idx_ret = rets.mean(axis=1)  # ligevægts-"indeks"-afkast
    sigma_i = rets.rolling(window=window, min_periods=window).std(ddof=1)
    sigma_p = idx_ret.rolling(window=window, min_periods=window).std(ddof=1)
    n = sigma_i.notna().sum(axis=1).replace(0, np.nan)
    sum_sig = sigma_i.sum(axis=1)
    sum_sig2 = (sigma_i ** 2).sum(axis=1)
    num = sigma_p ** 2 - sum_sig2 / (n ** 2)
    den = (sum_sig ** 2 - sum_sig2) / (n ** 2)
    corr = num / den.replace(0, np.nan)
    return corr.clip(lower=0.0, upper=1.0)


def build_macro_frame(cache_dir: Path | None = None, *, period: str = "max") -> pd.DataFrame:
    """Byg den markeds-brede makro-frame (date-index, kolonner = MACRO_FEATURE_COLUMNS).

    Kolonner hvis kilde fejler udelades stille (feature-laget fylder dem neutralt).
    """
    cdir = Path(cache_dir) if cache_dir is not None else Path(config.CACHE_DIR)
    cols: dict[str, pd.Series] = {}

    # --- VIX-termstruktur + vol-of-vol (yfinance) ---
    vix = _yf_close("^VIX", period)
    vix3m = _yf_close("^VIX3M", period)
    if vix is not None and vix3m is not None:
        slope = (vix / vix3m.reindex(vix.index).ffill()).replace([np.inf, -np.inf], np.nan)
        cols["vix_ts_slope"] = slope
    vvix = _yf_close("^VVIX", period)
    if vvix is not None:
        cols["vvix_level"] = vvix / 100.0

    # --- Kredit-proxy: HYG/LQD-forhold, 5d pct-ændring (negativ i kredit-stress) ---
    hyg = _yf_close("HYG", period)
    lqd = _yf_close("LQD", period)
    if hyg is not None and lqd is not None:
        ratio = hyg / lqd.reindex(hyg.index).ffill()
        cols["credit_ratio_chg"] = ratio.pct_change(periods=5)

    # --- Bond-vol: ^MOVE, 5d pct-ændring (positiv i bond-stress) ---
    move = _yf_close("^MOVE", period)
    if move is not None:
        cols["move_chg"] = move.pct_change(periods=5)

    # --- Olie (WTI front-month CL=F): log-afkast + annualiseret log-vol. Markeds-bred
    #     (relevant for hele markedet, især de ~19 energiselskaber). Samme kilde (yfinance)
    #     og form som ^VIX-familien / de stationære pris-features (log_ret, vol_annual_pct). ---
    oil = _yf_close("CL=F", period)
    if oil is not None:
        cols["oil_log_ret"] = np.log(oil / oil.shift(1)).replace([np.inf, -np.inf], np.nan)
        cols["oil_vol_annual_pct"] = rolling_annualized_log_vol_pct(oil)

    # --- Breadth + tværsnits-korrelation fra cachen (ingen ekstern kilde) ---
    closes = _watchlist_close_frame(cdir)
    if not closes.empty:
        cols["breadth_pct"] = _breadth(closes, int(config.MACRO_BREADTH_MA_DAYS))
        cols["xsec_corr"] = _xsec_corr(closes, int(config.MACRO_CORR_WINDOW_DAYS))

    if not cols:
        logger.warning("Ingen makro-kilder tilgængelige — tom frame.")
        return pd.DataFrame()

    frame = pd.DataFrame(cols).sort_index()
    frame.index = _to_naive_daily(frame.index)
    frame = frame[~frame.index.duplicated(keep="last")]
    # Behold kun de definerede kolonner i autoritativ rækkefølge (dem der findes).
    ordered = [c for c in config.MACRO_FEATURE_COLUMNS if c in frame.columns]
    missing = [c for c in config.MACRO_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        logger.warning("Makro-kolonner uden kilde (fyldes neutralt nedstrøms): %s", ", ".join(missing))
    return frame[ordered]


def save_macro_frame(frame: pd.DataFrame, path: Path | None = None) -> Path:
    dest = Path(path) if path is not None else Path(config.MACRO_CACHE_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_parquet(frame, dest)
    return dest


def load_macro_frame(path: Path | None = None) -> pd.DataFrame | None:
    """Indlæs den cachede makro-frame; None hvis den ikke findes/er tom."""
    src = Path(path) if path is not None else Path(config.MACRO_CACHE_PATH)
    if not src.is_file():
        return None
    try:
        df = pd.read_parquet(src)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke læse makro-cache %s: %s", src, exc)
        return None
    if df.empty:
        return None
    df.index = _to_naive_daily(df.index)
    return df.sort_index()


def build_and_cache_macro_frame(*, period: str = "max") -> pd.DataFrame:
    """Byg + gem makro-framen til config.MACRO_CACHE_PATH og returnér den."""
    frame = build_macro_frame(period=period)
    if frame.empty:
        return frame
    dest = save_macro_frame(frame)
    logger.info(
        "Makro-frame gemt: %s rækker × %s kolonner (%s..%s) -> %s",
        len(frame), frame.shape[1],
        frame.index.min().date(), frame.index.max().date(), dest,
    )
    return frame


def ensure_macro_oil_cache(*, force_rebuild: bool = False) -> None:
    """Sørg for at makro-/olie-kolonnerne er materialiseret i Parquet-cachen før træning.

    Kaldes i starten af train_model, så `--train` er én kommando: (1) genbyg den markeds-brede
    makro-frame (inkl. WTI CL=F) hvis den mangler/er forældet, (2) sæt den in-proces frame-cache
    så inkrementel tail-merge bruger den friske frame, (3) materialisér kolonnerne ind i hver
    ticker-parquet der mangler én eller flere af config.MACRO_FEATURE_COLUMNS (billig skema-tjek).
    Værdi-/dag-opdatering for friske barer sker via den normale inkrementelle tail-merge, så her
    materialiseres kun filer hvor kolonne-SÆTTET mangler (fx første kørsel efter at olie er tilføjet).
    No-op når MACRO_FEATURES_ENABLED er slået fra.
    """
    if not getattr(config, "MACRO_FEATURES_ENABLED", False):
        return

    from stock_predictor.data_fetcher import required_last_bar_date, set_macro_frame_cache

    frame = load_macro_frame()
    stale = (
        force_rebuild
        or frame is None
        or frame.empty
        or frame.index.max().date() < required_last_bar_date()
    )
    if stale:
        try:
            rebuilt = build_and_cache_macro_frame()
            if rebuilt is not None and not rebuilt.empty:
                frame = rebuilt
        except Exception as exc:  # noqa: BLE001
            logger.warning("Makro-/olie-frame genbygning fejlede (bruger gemt frame): %s", exc)

    if frame is None or frame.empty:
        logger.warning("Ingen makro-/olie-frame tilgængelig — springer cache-materialisering over.")
        return

    # Lad inkrementel tail-merge i denne proces bruge den friske frame.
    set_macro_frame_cache(frame)

    expected = set(config.MACRO_FEATURE_COLUMNS)
    cache_dir = Path(config.CACHE_DIR)
    files = sorted(cache_dir.glob("*.parquet"))
    todo: list[Path] = []
    for path in files:
        try:
            import pyarrow.parquet as pq

            names = set(pq.ParquetFile(path).schema.names)
        except Exception:  # noqa: BLE001 - fald tilbage til fuld læsning ved skema-fejl
            try:
                names = set(pd.read_parquet(path).columns)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Springer %s over (kan ikke læse skema): %s", path.name, exc)
                continue
        if not expected.issubset(names):
            todo.append(path)

    if not todo:
        logger.info("Makro-/olie-kolonner allerede i alle %s cache-filer.", len(files))
        return

    # Doven import (som main.py → tools.update_news_sentiment): undgår import-cyklus, da
    # tools.backfill_macro_features importerer fra dette modul ved load-tid.
    from stock_predictor.tools.backfill_macro_features import backfill_file

    logger.info("Materialiserer makro-/olie-kolonner i %s/%s cache-filer …", len(todo), len(files))
    ok = 0
    for path in todo:
        try:
            if backfill_file(path, frame) > 0:
                ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Materialisering fejlede for %s: %s", path.name, exc)
    logger.info("Makro-/olie-materialisering færdig: %s/%s filer opdateret.", ok, len(todo))


__all__ = [
    "build_macro_frame",
    "build_and_cache_macro_frame",
    "save_macro_frame",
    "load_macro_frame",
    "ensure_macro_oil_cache",
]
