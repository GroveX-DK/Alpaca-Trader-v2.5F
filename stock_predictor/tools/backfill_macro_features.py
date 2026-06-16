"""Backfill de markeds-brede makro-krise-kolonner ind i den eksisterende Parquet-cache.

Bygger (eller genbruger) den markeds-brede makro-frame (VIX-termstruktur, breadth,
tværsnits-korrelation, kreditspænd, bond-vol) og materialiserer dens kolonner ind i hver
tickers OHLCV-cache via build_dataset_frame — uden at re-downloade OHLCV. Rå makro-kolonner
gemmes som base (samme mønster som vix_close), så de bæres med ved senere inkrementel merge
og er klar når MACRO_FEATURES_ENABLED slås til + modellen genoptrænes.

Kør:
    python -m stock_predictor.tools.backfill_macro_features            # byg frame + alle filer
    python -m stock_predictor.tools.backfill_macro_features --use-cached-frame
    python -m stock_predictor.tools.backfill_macro_features --symbols-only AAPL MSFT
    python -m stock_predictor.tools.backfill_macro_features --max-tickers 3   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # Windows-konsol (cp1252) kan ikke alle tegn; tving UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import _atomic_save_parquet  # noqa: E402
from stock_predictor.feature_engineer import FEATURE_COLUMNS, build_dataset_frame  # noqa: E402
from stock_predictor.macro_features import (  # noqa: E402
    build_and_cache_macro_frame,
    load_macro_frame,
)

_OHLCV = ("open", "high", "low", "close", "volume")
_MACRO_COLS = tuple(getattr(config, "MACRO_FEATURE_COLUMNS", ()))

# Forældede feature-navne fra tidligere featuresæt (droppes ved genberegning).
_LEGACY_FEATURE_COLUMNS = (
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "ema_20", "ema_50", "obv",
)
# Drop afledte tekniske features før rebuild; bevar eksterndata (vix/news/makro),
# som build_dataset_frame enten bærer med (base) eller overskriver fra makro-framen.
_DROP_BEFORE_REBUILD = (
    (set(FEATURE_COLUMNS) - {"vix_close", "news_sentiment"} - set(_MACRO_COLS))
    | set(_LEGACY_FEATURE_COLUMNS)
)


def _load_raw(path: Path) -> pd.DataFrame | None:
    df = pd.read_parquet(path)
    if df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if not set(_OHLCV).issubset(df.columns):
        return None
    return df


def backfill_file(path: Path, macro: pd.DataFrame) -> int:
    df = _load_raw(path)
    if df is None:
        return 0
    vix = pd.to_numeric(df["vix_close"], errors="coerce") if "vix_close" in df.columns else None
    base = df.drop(columns=[c for c in df.columns if c in _DROP_BEFORE_REBUILD], errors="ignore")
    for c in _OHLCV:
        base[c] = base[c].astype(float)
    full = build_dataset_frame(base, vix, macro)
    _atomic_save_parquet(full, path)
    return len(full)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill makro-krise-kolonner i Parquet-cachen.")
    p.add_argument("--symbols-only", nargs="*", help="Kun disse tickere (default: alle cache-filer).")
    p.add_argument("--max-tickers", type=int, default=0, help="Begræns antal (smoke-test).")
    p.add_argument("--cache-dir", default=None, help="Override cache-mappe (default: config.CACHE_DIR).")
    p.add_argument("--use-cached-frame", action="store_true",
                   help="Genbrug gemt makro-frame i stedet for at bygge en ny (ingen yfinance-kald).")
    p.add_argument("--period", default="max", help="yfinance-periode ved frame-bygning (default: max).")
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(config.CACHE_DIR).resolve()
    if not cache_dir.is_dir():
        print(f"Cache-mappe findes ikke: {cache_dir}", flush=True)
        return 2

    if args.use_cached_frame:
        macro = load_macro_frame()
        if macro is None or macro.empty:
            print("Ingen gemt makro-frame fundet — kør uden --use-cached-frame for at bygge.", flush=True)
            return 2
        print(f"Genbruger gemt makro-frame: {len(macro)} rækker × {macro.shape[1]} kolonner.", flush=True)
    else:
        print("Bygger makro-frame (yfinance + cache-breadth/korrelation) …", flush=True)
        macro = build_and_cache_macro_frame(period=args.period)
        if macro.empty:
            print("Makro-frame blev tom — afbryder (tjek yfinance/netværk).", flush=True)
            return 2
        print(f"Makro-frame: {len(macro)} rækker × {macro.shape[1]} kolonner "
              f"({', '.join(macro.columns)}).", flush=True)

    if args.symbols_only:
        wanted = [s.strip().upper() for s in args.symbols_only if s.strip()]
        files = [cache_dir / f"{s}.parquet" for s in wanted]
        files = [f for f in files if f.is_file()]
    else:
        files = sorted(cache_dir.glob("*.parquet"))
    if args.max_tickers and args.max_tickers > 0:
        files = files[: args.max_tickers]

    print(f"Backfiller {len(files)} fil(er) i {cache_dir}.", flush=True)
    ok = 0
    failed: list[str] = []
    for i, f in enumerate(files, 1):
        sym = f.stem
        try:
            n = backfill_file(f, macro)
            if n > 0:
                ok += 1
                print(f"[{i}/{len(files)}] OK {sym}: {n} rækker", flush=True)
            else:
                failed.append(sym)
                print(f"[{i}/{len(files)}] SKIP {sym}: tom / mangler OHLCV", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(sym)
            print(f"[{i}/{len(files)}] FEJL {sym}: {exc}", flush=True)

    print(f"\nFærdig: {ok}/{len(files)} backfillet.", flush=True)
    if failed:
        print(f"Mislykkedes ({len(failed)}): {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
