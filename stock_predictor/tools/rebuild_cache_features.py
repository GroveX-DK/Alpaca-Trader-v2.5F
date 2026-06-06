"""Genberegn feature-kolonner i den eksisterende Parquet-cache *uden* at re-downloade.

Bruges når feature-definitionerne ændres (fx OHLCV → stationære/skalainvariante features).
Pr. fil bevares rå OHLCV + vix_close (kilde-/eksterndata) og evt. Watchlist-metriker;
forældede feature-kolonner droppes, og hele FEATURE_COLUMNS genberegnes via
build_dataset_frame (vol_annual_pct + 21 features), med fuld historik (NaN-opvarmning).

Kør:
    python -m stock_predictor.tools.rebuild_cache_features
    python -m stock_predictor.tools.rebuild_cache_features --symbols-only AAPL MSFT
    python -m stock_predictor.tools.rebuild_cache_features --max-tickers 3   # smoke
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

_OHLCV = ("open", "high", "low", "close", "volume")

# Forældede feature-navne fra tidligere featuresæt der skal droppes ved genberegning.
_LEGACY_FEATURE_COLUMNS = (
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "ema_20", "ema_50", "obv",
)

# Kolonner der ikke skal bæres med ind i build_dataset_frame (genberegnes derinde).
# vix_close beholdes (eksterndata) og sendes som parameter.
_DROP_BEFORE_REBUILD = (set(FEATURE_COLUMNS) - {"vix_close"}) | set(_LEGACY_FEATURE_COLUMNS)


def _load_raw(path: Path) -> pd.DataFrame | None:
    """Læs hele Parquet-filen med datetime-index (bevarer alle kolonner)."""
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


def rebuild_file(path: Path) -> int:
    """Genberegn én cache-fil. Returnerer antal rækker skrevet (0 ved fejl/skip)."""
    df = _load_raw(path)
    if df is None:
        return 0

    vix = None
    if "vix_close" in df.columns:
        vix = pd.to_numeric(df["vix_close"], errors="coerce")

    # Behold OHLCV + evt. Watchlist-metriker; drop forældede/afledte feature-kolonner.
    base = df.drop(columns=[c for c in df.columns if c in _DROP_BEFORE_REBUILD], errors="ignore")
    for c in _OHLCV:
        base[c] = base[c].astype(float)

    full = build_dataset_frame(base, vix)
    _atomic_save_parquet(full, path)
    return len(full)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Genberegn feature-kolonner i Parquet-cachen.")
    p.add_argument("--symbols-only", nargs="*", help="Kun disse tickere (default: alle filer i cachen).")
    p.add_argument("--max-tickers", type=int, default=0, help="Begræns antal (smoke-test).")
    p.add_argument("--cache-dir", default=None, help="Override cache-mappe (default: config.CACHE_DIR).")
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(config.CACHE_DIR).resolve()
    if not cache_dir.is_dir():
        print(f"Cache-mappe findes ikke: {cache_dir}", flush=True)
        return 2

    if args.symbols_only:
        wanted = [s.strip().upper() for s in args.symbols_only if s.strip()]
        files = [cache_dir / f"{s}.parquet" for s in wanted]
        files = [f for f in files if f.is_file()]
    else:
        files = sorted(cache_dir.glob("*.parquet"))
    if args.max_tickers and args.max_tickers > 0:
        files = files[: args.max_tickers]

    print(f"Genberegner {len(files)} fil(er) i {cache_dir}", flush=True)
    print(f"Nye feature-kolonner ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}", flush=True)

    ok = 0
    failed: list[str] = []
    for i, f in enumerate(files, 1):
        sym = f.stem
        try:
            n = rebuild_file(f)
            if n > 0:
                ok += 1
                print(f"[{i}/{len(files)}] OK {sym}: {n} rækker", flush=True)
            else:
                failed.append(sym)
                print(f"[{i}/{len(files)}] SKIP {sym}: tom / mangler OHLCV", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(sym)
            print(f"[{i}/{len(files)}] FEJL {sym}: {exc}", flush=True)

    print(f"\nFærdig: {ok}/{len(files)} genberegnet.", flush=True)
    if failed:
        print(f"Mislykkedes ({len(failed)}): {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
