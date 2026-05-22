"""Én-gangs (eller gentagen) import: {SYMBOL}_historical.csv -> cache/{SYMBOL}.parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from stock_predictor import config  # noqa: E402
from stock_predictor.feature_engineer import rolling_annualized_log_vol_pct  # noqa: E402


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[-1]).strip().lower() for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _price_series_for_vol(df: pd.DataFrame) -> pd.Series:
    """Justeret/split-korrigeret pris til log-afkast hvis muligt, ellers lukkekurs."""
    if "adj close" in df.columns:
        return df["adj close"].astype(float)
    if "adj_close" in df.columns:
        return df["adj_close"].astype(float)
    if "price" in df.columns:
        return df["price"].astype(float)
    return df["close"].astype(float)


def _csv_to_df_flat(csv_path: Path) -> pd.DataFrame:
    """Én header-række: Date, Open, High, ... eller Date, Price, Close, ... (fx yfinance)."""
    df = _lower_columns(pd.read_csv(csv_path))
    if "date" not in df.columns:
        raise ValueError("CSV mangler datokolonne (forventet 'Date' / 'date').")
    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV mangler OHLCV-kolonner; har: {sorted(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    prices = _price_series_for_vol(df).reindex(df.index)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    prices.index = out.index
    out["vol_annual_pct"] = rolling_annualized_log_vol_pct(prices)
    return out


def _csv_to_df_yahoo_skip3(csv_path: Path) -> pd.DataFrame:
    # Række 1–3: Yahoo-web metadata; data fra række 4. Typisk: Date, Open, High, Low, Close, [Adj Close,] Volume
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        line1 = f.readline()
        line2 = f.readline()
    has_adj = "Adj Close" in line1 or "Adj Close" in line2
    if has_adj:
        names = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    else:
        names = ["date", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(csv_path, skiprows=3, names=names)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    prices = _price_series_for_vol(df).reindex(df.index)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = pd.to_datetime(out.index).normalize().tz_localize(None)
    prices.index = out.index
    out["vol_annual_pct"] = rolling_annualized_log_vol_pct(prices)
    return out


def _looks_like_yahoo_skip3(csv_path: Path) -> bool:
    """True hvis første linje ikke ser ud som et almindeligt data-header (fx 'Ticker,...')."""
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        first = f.readline().strip().lower()
    if first.startswith("date,"):
        return False
    return True


def _csv_to_df(csv_path: Path) -> pd.DataFrame:
    if _looks_like_yahoo_skip3(csv_path):
        return _csv_to_df_yahoo_skip3(csv_path)
    return _csv_to_df_flat(csv_path)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Importer historiske CSV til OHLCV-cache.")
    p.add_argument(
        "--delete-csv",
        action="store_true",
        help="Slet {SYM}_historical.csv efter succesfuld skrivning.",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=_ROOT,
        help="Rodmappe hvor SYM_historical.csv ligger.",
    )
    args = p.parse_args()
    root: Path = args.project_root.resolve()
    cache_dir = Path(config.CACHE_DIR).resolve()

    ok = 0
    had_err = False
    missing_csv: list[str] = []
    for sym in config.WATCHLIST:
        sym = sym.strip().upper()
        csv_path = root / f"{sym}_historical.csv"
        if not csv_path.is_file():
            missing_csv.append(sym)
            continue
        try:
            df = _csv_to_df(csv_path)
            if df.empty:
                print(f"WARN: empty after parse: {sym} (remove stale parquet if any)", flush=True)
                stale = cache_dir / f"{sym}.parquet"
                stale.unlink(missing_ok=True)
                continue
            dest = cache_dir / f"{sym}.parquet"
            _atomic_parquet(df, dest)
            print(f"OK {sym}: {len(df)} rows -> {dest.name}", flush=True)
            ok += 1
            if args.delete_csv:
                csv_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            had_err = True
            print(f"ERR {sym}: {exc}", flush=True)

    if missing_csv:
        print(f"Missing CSV ({len(missing_csv)}): {', '.join(missing_csv)}", flush=True)

    # Remove stale parquet files (symbols not on watchlist)
    watch = {s.strip().upper() for s in config.WATCHLIST}
    removed = 0
    if cache_dir.is_dir():
        for pq in cache_dir.glob("*.parquet"):
            stem = pq.stem.upper()
            if stem not in watch:
                pq.unlink(missing_ok=True)
                removed += 1
                print(f"Removed stale cache: {pq.name}", flush=True)
    if removed:
        print(f"Removed {removed} stale .parquet file(s).", flush=True)

    print(f"Done: {ok} symbols written to {cache_dir}", flush=True)
    return 1 if had_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
