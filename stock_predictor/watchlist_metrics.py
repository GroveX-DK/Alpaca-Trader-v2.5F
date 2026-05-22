"""Ekstra daglige metriker (Watchlist-CSV-kompatibel) ud fra OHLCV."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

# Parkinson: sqrt(gennemsnit af k·ln(H/L)²) over vindue, annualiseret med √252.
# Kalibreret mod output/Watchlist/AAPL.csv (vindue 20, min_periods=1).
_PARKINSON_K: Final[float] = 1.0 / (4.0 * math.log(2.0))
PARKINSON_WINDOW: Final[int] = 20
TRADING_DAYS_PER_YEAR: Final[int] = 252

WATCHLIST_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "daily_return",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "parkinson_vol",
    "atr_14",
)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_watchlist_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tilføj/overskriv Watchlist-kolonner ud fra open, high, low, close.

    Definitioner (matcher AAPL Watchlist-CSV inden for lille tolerance):
    - daily_return: ln(close / close.shift(1)) (log-afkast, som Watchlist-CSV).
    - rolling_vol_20d / rolling_vol_60d: std af log-afkast, rullende min_periods=1, ×√252.
      rullende med min_periods=1, ×√252 (annualiseret som decimal, ikke ×100).
    - parkinson_vol: √(rolling mean af k·ln(H/L)²), k=1/(4 ln 2)), vindue 20,
      min_periods=1, ×√252.
    - atr_14: rullende gennemsnit af true range, vindue 14, min_periods=1
      (SMA af TR — ikke Wilder-RMA).
    """
    out = df.sort_index().copy()
    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(out.columns):
        raise ValueError(f"Mangler OHLCV-kolonner; har {sorted(out.columns)}")

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    # Watchlist-CSV: daily_return = ln(C_t / C_{t-1}), ikke simpelt pct_change.
    log_r = np.log(close / close.shift(1))
    out["daily_return"] = log_r

    out["rolling_vol_20d"] = (
        log_r.rolling(window=20, min_periods=1).std(ddof=1)
        * np.sqrt(float(TRADING_DAYS_PER_YEAR))
    )
    out["rolling_vol_60d"] = (
        log_r.rolling(window=60, min_periods=1).std(ddof=1)
        * np.sqrt(float(TRADING_DAYS_PER_YEAR))
    )

    hl = np.log(high / low.replace(0, np.nan))
    park_var = _PARKINSON_K * (hl**2)
    out["parkinson_vol"] = (
        np.sqrt(park_var.rolling(window=PARKINSON_WINDOW, min_periods=1).mean())
        * np.sqrt(float(TRADING_DAYS_PER_YEAR))
    )

    tr = _true_range(high, low, close)
    out["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    for c in WATCHLIST_METRIC_COLUMNS:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).astype(float)

    return out


def assert_watchlist_metrics_match_csv(
    csv_path: str,
    *,
    n_rows: int = 250,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Smoke: første n_rows med gyldige CSV-værdier skal matche compute_watchlist_metrics."""
    from pathlib import Path as _Path

    p = _Path(csv_path)
    raw = pd.read_csv(p)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()

    ohlcv = raw[["open", "high", "low", "close", "volume"]].astype(float)
    calc = compute_watchlist_metrics(ohlcv)

    checked = 0
    for col in WATCHLIST_METRIC_COLUMNS:
        if col not in raw.columns:
            raise AssertionError(f"CSV mangler kolonne {col}")
        for ts in calc.index[:n_rows]:
            exp = raw.loc[ts, col]
            got = calc.loc[ts, col]
            if pd.isna(exp) and pd.isna(got):
                continue
            if pd.isna(exp) or pd.isna(got):
                continue
            checked += 1
            if not np.isclose(float(got), float(exp), rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{col} @ {ts.date()}: forventet {exp}, fik {got} (rtol={rtol})",
                )
    if checked == 0:
        raise AssertionError("Ingen sammenlignelige ikke-NaN celler i det valgte vindue.")
