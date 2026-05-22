"""Tekniske indikatorer pr. aktie-pr. dag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor import config
from stock_predictor.watchlist_metrics import WATCHLIST_METRIC_COLUMNS, compute_watchlist_metrics


def rolling_annualized_log_vol_pct(
    closes: pd.Series,
    n_returns: int | None = None,
    trading_days_per_year: int | None = None,
) -> pd.Series:
    """
    Pr. dato: std (ddof=1) over de sidste n_returns daglige log-afkast r_t = ln(P_t / P_{t-1}),
    annualiseret som σ * sqrt(252) * 100 (procentpoint). Første række(r) uden nok historik: NaN.
    """
    n = int(config.VOLATILITY_ROLLING_RETURNS if n_returns is None else n_returns)
    td = int(config.VOLATILITY_TRADING_DAYS_PER_YEAR if trading_days_per_year is None else trading_days_per_year)
    s = closes.astype(float).sort_index()
    log_ret = np.log(s / s.shift(1))
    daily_sigma = log_ret.rolling(window=n, min_periods=n).std(ddof=1)
    return (daily_sigma * np.sqrt(float(td)) * 100.0).reindex(s.index)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(macd_line, signal)
    return macd_line, sig


def bollinger_bandwidth_pct(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(window=window).mean()
    sd = close.rolling(window=window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = upper - lower
    pos = (close - lower) / width.replace(0, np.nan)
    return pos


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: OHLCV med index = dato, kolonner open,high,low,close,volume
    (+ valgfrit forudberegnede Watchlist-kolonner — genberegnes her fra OHLCV).
    Output: 15 features; NaN før indikatorer er varme.
    """
    df = df.sort_index().copy()
    close = df["close"]
    vol = df["volume"]

    daily_return_pct = close.pct_change() * 100.0
    out = pd.DataFrame(index=df.index)
    out["daily_return_pct"] = daily_return_pct
    out["ema_5"] = ema(close, 5)
    out["ema_10"] = ema(close, 10)
    out["ema_20"] = ema(close, 20)
    out["rsi_14"] = rsi(close, 14)
    macd_l, sig = macd_signal(close)
    out["macd"] = macd_l
    out["macd_signal"] = sig
    out["bb_position"] = bollinger_bandwidth_pct(close)
    ma_vol_10 = vol.rolling(window=10).mean()
    out["volume_rel_10"] = vol / ma_vol_10.replace(0, np.nan)
    out["ann_vol_log_pct"] = rolling_annualized_log_vol_pct(close)

    wm = compute_watchlist_metrics(df)
    for col in WATCHLIST_METRIC_COLUMNS:
        out[col] = wm[col]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    out = out.dropna()
    return out


def targets_next_day_open_to_close_pct(ohlcv: pd.DataFrame, idx: pd.Index) -> pd.Series:
    """Næste handelsdags intradag-afkast (open→close) i pct, alignet på dag t før hop."""
    open_ = ohlcv["open"].reindex(idx).astype(float)
    close = ohlcv["close"].reindex(idx).astype(float)
    nxt_open = open_.shift(-1)
    nxt_close = close.shift(-1)
    return (nxt_close / nxt_open.replace(0, np.nan) - 1.0) * 100.0
