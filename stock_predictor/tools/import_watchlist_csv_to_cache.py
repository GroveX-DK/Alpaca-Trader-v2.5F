"""Importer output/Watchlist/{SYM}.csv til stock_predictor/cache/{SYM}.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import _atomic_save_parquet  # noqa: E402
from stock_predictor.feature_engineer import rolling_annualized_log_vol_pct  # noqa: E402
from stock_predictor.watchlist_metrics import (  # noqa: E402
    WATCHLIST_METRIC_COLUMNS,
    compute_watchlist_metrics,
)


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def watchlist_csv_to_ohlcv_frame(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Læs Watchlist-format: Ticker, Date, OHLCV (+ valgfri ekstrakolonner)."""
    raw = _lower_columns(pd.read_csv(csv_path))
    if "date" not in raw.columns:
        raise ValueError(f"Mangler Date-kolonne i {csv_path}")
    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(raw.columns):
        raise ValueError(f"Mangler OHLCV i {csv_path}; har {sorted(raw.columns)}")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw.index = pd.to_datetime(raw.index).normalize().tz_localize(None)
    ohlcv = raw[list(need)].astype(float)
    return raw, ohlcv


def build_cache_dataframe(raw: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Foretræk CSV-værdier for Watchlist-metriker hvor de findes; ellers beregnet."""
    calc = compute_watchlist_metrics(ohlcv)
    merged = calc.copy()
    for col in WATCHLIST_METRIC_COLUMNS:
        if col not in raw.columns:
            continue
        csv_s = pd.to_numeric(raw[col], errors="coerce")
        merged[col] = csv_s.where(csv_s.notna(), merged[col])

    price = ohlcv["close"].astype(float)
    if "adj_close" in raw.columns:
        ac = pd.to_numeric(raw["adj_close"], errors="coerce")
        price = ac.where(ac.notna(), price)
    merged["vol_annual_pct"] = rolling_annualized_log_vol_pct(price)
    return merged


def main() -> int:
    p = argparse.ArgumentParser(
        description="Konverter Watchlist-CSV til OHLCV+metriker Parquet-cache.",
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=config.WATCHLIST_CSV_DIR,
        help=f"Mappe med {{SYM}}.csv (default: {config.WATCHLIST_CSV_DIR})",
    )
    p.add_argument(
        "--symbols-only",
        nargs="*",
        help="Kun disse tickere (STORE BOGSTAVER); default: alle *.csv i mappen.",
    )
    args = p.parse_args()
    watch_dir: Path = args.dir.resolve()
    cache_dir = Path(config.CACHE_DIR).resolve()

    if not watch_dir.is_dir():
        print(f"FEJL: mappen findes ikke: {watch_dir}", flush=True)
        return 1

    if args.symbols_only:
        files = [watch_dir / f"{s.strip().upper()}.csv" for s in args.symbols_only]
    else:
        files = sorted(watch_dir.glob("*.csv"))

    ok = 0
    err = False
    for csv_path in files:
        if not csv_path.is_file():
            print(f"SPRING OVER (mangler fil): {csv_path.name}", flush=True)
            err = True
            continue
        sym = csv_path.stem.strip().upper()
        try:
            raw, ohlcv = watchlist_csv_to_ohlcv_frame(csv_path)
            if ohlcv.empty:
                print(f"TOM: {sym}", flush=True)
                err = True
                continue
            full = build_cache_dataframe(raw, ohlcv)
            dest = cache_dir / f"{sym}.parquet"
            _atomic_save_parquet(full, dest)
            print(f"OK {sym}: {len(full)} rækker -> {dest.name}", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            err = True
            print(f"FEJL {sym}: {exc}", flush=True)

    print(f"Færdig: {ok} symbol(er) skrevet til {cache_dir}", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
