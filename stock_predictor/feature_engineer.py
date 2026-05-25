"""Tekniske indikatorer, kalender- og makrofeatures pr. aktie-pr. dag.

Featuresæt (21 kolonner, beregnes fra OHLCV med ren pandas + ^VIX):

Tekniske indikatorer:
    rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_lower, bb_width,
    ema_20, ema_50, obv, stoch_k, cci_20
Kalender (fra dato-index):
    day_of_week, month
Makro (fra yfinance ^VIX, gemt som kolonne vix_close i cache):
    vix_close
Rå OHLCV + volatilitet (alle kolonner fra cache-parquet):
    open, high, low, close, volume, vol_annual_pct
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stock_predictor import config

# Rækkefølge = kolonnerækkefølge ud af engineer_features (skal matche N_FEATURES).
FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "ema_20",
    "ema_50",
    "obv",
    "stoch_k",
    "cci_20",
    "day_of_week",
    "month",
    "vix_close",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vol_annual_pct",
)


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


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Øvre/nedre Bollinger-bånd (absolut prisniveau) + bredde (øvre − nedre)."""
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    width = upper - lower
    return upper, lower, width


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: kumulativt volumen med fortegn af daglig prisændring."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.astype(float)).fillna(0.0).cumsum()


def stochastic_k(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Stochastic Oscillator %K = 100·(close − min_low) / (max_high − min_low)."""
    ll = low.rolling(window=period, min_periods=period).min()
    hh = high.rolling(window=period, min_periods=period).max()
    rng = (hh - ll).replace(0, np.nan)
    return 100.0 * (close - ll) / rng


def _rolling_mad(values: pd.Series, window: int) -> pd.Series:
    """Mean absolute deviation omkring vinduets eget gennemsnit (vektoriseret)."""
    arr = values.to_numpy(dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.shape[0] >= window:
        win = sliding_window_view(arr, window)
        means = win.mean(axis=1, keepdims=True)
        mad = np.abs(win - means).mean(axis=1)
        out[window - 1 :] = mad
    return pd.Series(out, index=values.index)


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index = (TP − SMA(TP)) / (0.015 · MAD(TP)), TP = (H+L+C)/3."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=period, min_periods=period).mean()
    mad = _rolling_mad(tp, period)
    denom = (0.015 * mad).replace(0, np.nan)
    return (tp - sma) / denom


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Alle 21 feature-kolonner (uden dropna), index = dato. Kræver OHLCV (+ valgfri vix_close)."""
    df = df.sort_index().copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi(close, 14)
    macd_l, sig = macd_signal(close)
    out["macd"] = macd_l
    out["macd_signal"] = sig
    out["macd_hist"] = macd_l - sig
    up, lo, width = bollinger_bands(close)
    out["bb_upper"] = up
    out["bb_lower"] = lo
    out["bb_width"] = width
    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["obv"] = obv(close, vol)
    out["stoch_k"] = stochastic_k(high, low, close, 14)
    out["cci_20"] = cci(high, low, close, 20)

    idx = pd.DatetimeIndex(df.index)
    out["day_of_week"] = idx.dayofweek.astype(float)
    out["month"] = idx.month.astype(float)

    if "vix_close" in df.columns:
        out["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce").ffill()
    else:
        out["vix_close"] = np.nan

    # Rå OHLCV som features (absolutte niveauer — standardiseres ved træning).
    out["open"] = df["open"].astype(float)
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["volume"] = vol
    if "vol_annual_pct" in df.columns:
        out["vol_annual_pct"] = pd.to_numeric(df["vol_annual_pct"], errors="coerce")
    else:
        out["vol_annual_pct"] = rolling_annualized_log_vol_pct(close)

    return out[list(FEATURE_COLUMNS)]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: OHLCV med index = dato, kolonner open,high,low,close,volume (+ valgfri vix_close).
    Output: 21 features (FEATURE_COLUMNS); rækker droppes mens indikatorer er kolde eller
    vix_close mangler.
    """
    out = _feature_frame(df)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def build_dataset_frame(ohlcv: pd.DataFrame, vix_close: pd.Series | None = None) -> pd.DataFrame:
    """
    Fuldt datasæt til cache: OHLCV + vol_annual_pct + vix_close + alle 21 features
    (uden dropna, så fuld historik bevares med NaN-opvarmning).
    """
    base = ohlcv.sort_index().copy()
    if vix_close is not None:
        vix = pd.to_numeric(vix_close, errors="coerce")
        base["vix_close"] = vix.reindex(base.index).ffill()
    base["vol_annual_pct"] = rolling_annualized_log_vol_pct(base["close"])
    feats = _feature_frame(base)
    # Undgå dublerede kolonner (vix_close findes både i base og feats).
    extra = feats.drop(columns=[c for c in feats.columns if c in base.columns], errors="ignore")
    return pd.concat([base, extra], axis=1)


def targets_next_day_open_to_close_pct(ohlcv: pd.DataFrame, idx: pd.Index) -> pd.Series:
    """Næste handelsdags intradag-afkast (open→close) i pct, alignet på dag t før hop."""
    open_ = ohlcv["open"].reindex(idx).astype(float)
    close = ohlcv["close"].reindex(idx).astype(float)
    nxt_open = open_.shift(-1)
    nxt_close = close.shift(-1)
    return (nxt_close / nxt_open.replace(0, np.nan) - 1.0) * 100.0
