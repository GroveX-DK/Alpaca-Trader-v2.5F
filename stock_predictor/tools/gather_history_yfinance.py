"""Hent fuld historik for hele WATCHLIST via yfinance og gem som beriget Parquet-cache.

Pr. ticker gemmes: OHLCV (auto-justeret) + vol_annual_pct + vix_close + alle
feature-kolonner (RSI, MACD/hist, Bollinger, EMA20/50, OBV, Stoch %K, CCI, kalender).
^VIX hentes én gang og joines på hver tickers handelsdage (ffill).

Kør:
    python -m stock_predictor.tools.gather_history_yfinance
    python -m stock_predictor.tools.gather_history_yfinance --symbols-only AAPL MSFT
    python -m stock_predictor.tools.gather_history_yfinance --max-tickers 5   # smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # Windows-konsol (cp1252) kan ikke alle tegn; tving UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import _atomic_save_parquet  # noqa: E402
from stock_predictor.feature_engineer import build_dataset_frame  # noqa: E402

_OHLCV = ["open", "high", "low", "close", "volume"]


def _to_naive_daily(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx)
    if getattr(out, "tz", None) is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _extract_symbol(data: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    """Træk én tickers OHLCV ud af et (evt. multi-index) yfinance-resultat."""
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        lvl1 = data.columns.get_level_values(1)
        if sym in set(lvl0):
            sub = data[sym].copy()
        elif sym in set(lvl1):
            sub = data.xs(sym, axis=1, level=1).copy()
        else:
            return None
    else:
        sub = data.copy()

    sub.columns = [str(c).strip().lower() for c in sub.columns]
    if not set(_OHLCV).issubset(sub.columns):
        return None
    sub = sub[_OHLCV].copy()
    sub.index = _to_naive_daily(sub.index)
    sub = sub[~sub.index.duplicated(keep="last")].sort_index()
    sub = sub.dropna(how="all")
    if sub.empty:
        return None
    return sub.astype(float)


def fetch_vix_close(period: str = "max") -> pd.Series:
    raw = yf.download("^VIX", period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    vix = _extract_symbol(raw, "^VIX")
    if vix is None or vix.empty:
        raise RuntimeError("Kunne ikke hente ^VIX fra yfinance.")
    return vix["close"].rename("vix_close")


def _download_batch(symbols: list[str], period: str) -> pd.DataFrame:
    return yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def gather(
    symbols: list[str],
    cache_dir: Path,
    vix_close: pd.Series,
    *,
    period: str,
    batch_size: int,
    sleep_between: float,
) -> tuple[int, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed: list[str] = []

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        print(f"[{start + 1}-{start + len(batch)}/{len(symbols)}] henter {', '.join(batch)}",
              flush=True)
        try:
            data = _download_batch(batch, period)
        except Exception as exc:  # noqa: BLE001
            print(f"  BATCH-FEJL: {exc}", flush=True)
            failed.extend(batch)
            continue

        for sym in batch:
            try:
                ohlcv = _extract_symbol(data, sym)
                if ohlcv is None or ohlcv.empty:
                    print(f"  TOM: {sym}", flush=True)
                    failed.append(sym)
                    continue
                full = build_dataset_frame(ohlcv, vix_close)
                dest = cache_dir / f"{sym}.parquet"
                _atomic_save_parquet(full, dest)
                print(f"  OK {sym}: {len(full)} rækker "
                      f"({full.index.min().date()}->{full.index.max().date()})",
                      flush=True)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FEJL {sym}: {exc}", flush=True)
                failed.append(sym)

        if start + batch_size < len(symbols) and sleep_between > 0:
            time.sleep(sleep_between)

    return ok, failed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hent fuld historik via yfinance til Parquet-cache.")
    p.add_argument("--symbols-only", nargs="*", help="Kun disse tickere (default: hele WATCHLIST).")
    p.add_argument("--max-tickers", type=int, default=0, help="Begræns antal (smoke-test).")
    p.add_argument("--batch-size", type=int, default=25, help="Tickere pr. yfinance-kald.")
    p.add_argument("--period", default="max", help="yfinance-periode (default: max).")
    p.add_argument("--sleep", type=float, default=1.0, help="Sekunders pause mellem batches.")
    args = p.parse_args(argv)

    symbols = [s.strip().upper() for s in (args.symbols_only or config.WATCHLIST) if s.strip()]
    # Bevar rækkefølge, fjern dubletter.
    symbols = list(dict.fromkeys(symbols))
    if args.max_tickers and args.max_tickers > 0:
        symbols = symbols[: args.max_tickers]

    cache_dir = Path(config.CACHE_DIR).resolve()
    print(f"Henter ^VIX ...", flush=True)
    vix_close = fetch_vix_close(period=args.period)
    print(f"^VIX: {len(vix_close)} rækker "
          f"({vix_close.index.min().date()}->{vix_close.index.max().date()})", flush=True)

    ok, failed = gather(
        symbols, cache_dir, vix_close,
        period=args.period, batch_size=args.batch_size, sleep_between=args.sleep,
    )

    print(f"\nFærdig: {ok}/{len(symbols)} skrevet til {cache_dir}", flush=True)
    if failed:
        print(f"Mislykkedes ({len(failed)}): {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
