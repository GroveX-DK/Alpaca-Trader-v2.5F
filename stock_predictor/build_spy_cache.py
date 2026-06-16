"""Byg/refresh den lokale SPY-benchmark-cache (cache/SPY.parquet) fra yfinance.

Alpacas gratis IEX-feed har kun SPY fra ~2021, så backtest-benchmarken mangler før det.
yfinance leverer SPY tilbage til 2015. Vi henter splitter-/udbytte-justerede barer
(auto_adjust=True → intern konsistent OHLCV) og skriver dem i samme Parquet-skema som
resten af OHLCV-cachen (kolonner: open/high/low/close/volume, DatetimeIndex 'date').

Kør: ``python -m stock_predictor.build_spy_cache``
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from stock_predictor import config
from stock_predictor.data_fetcher import _atomic_save_parquet, _cache_path

logger = logging.getLogger(__name__)

_SYMBOL = "SPY"
_START = "2015-01-01"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance giver for ét ticker MultiIndex-kolonner ('Close','SPY') → fladgør til 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) for c in df.columns]
    return df


def build_spy_cache() -> pd.DataFrame:
    """Hent SPY (justeret) fra yfinance og skriv cache/SPY.parquet. Returnér den gemte frame."""
    end = date.today() + timedelta(days=1)  # yfinance `end` er eksklusiv
    raw = yf.download(
        _SYMBOL,
        start=_START,
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returnerede ingen SPY-data.")

    raw = _flatten_columns(raw)
    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    missing = [k for k in rename if k not in raw.columns]
    if missing:
        raise RuntimeError(f"yfinance-data mangler kolonner {missing}; fik {list(raw.columns)}")

    df = raw.rename(columns=rename)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="any")
    for col in df.columns:
        df[col] = df[col].astype(float)

    path = _cache_path(config.CACHE_DIR, _SYMBOL)
    _atomic_save_parquet(df, path)
    logger.info(
        "Skrev SPY-cache: %s barer (%s .. %s) → %s",
        len(df), df.index.min().date(), df.index.max().date(), path,
    )
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_spy_cache()
    print(f"SPY cache: {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")
    print(f"-> {_cache_path(config.CACHE_DIR, _SYMBOL)}")


if __name__ == "__main__":
    main()
