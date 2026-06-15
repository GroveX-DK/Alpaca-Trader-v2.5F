"""Tekniske indikatorer, kalender- og makrofeatures pr. aktie-pr. dag.

Featuresæt (21 kolonner, beregnes fra OHLCV med ren pandas + ^VIX). Alle pris-
afledte features er gjort *stationære / skalainvariante* (ændring frem for niveau),
så modellen generaliserer på tværs af aktier og tidsperioder uden data-leakage fra
absolutte prisniveauer.

Tekniske indikatorer (skalainvariante):
    rsi_14, macd_norm, macd_signal_norm, macd_hist_norm, bb_upper_rel, bb_lower_rel,
    bb_width_rel, ema_20_rel, ema_50_rel, obv_delta, stoch_k, cci_20
Kalender (fra dato-index):
    day_of_week, month
Makro (fra yfinance ^VIX, gemt som kolonne vix_close i cache):
    vix_close
Afkast / bar-form / volatilitet (afledt af OHLCV):
    log_ret = ln(C_t/C_{t-1}), high_ratio = H/O, low_ratio = L/O, close_ratio = C/O,
    vol_delta = ln(V_t/V_{t-1}), vol_annual_pct
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stock_predictor import config

# Basis-feature-kolonner (rækkefølge = kolonnerækkefølge ud af engineer_features).
_BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    "macd_norm",
    "macd_signal_norm",
    "macd_hist_norm",
    "bb_upper_rel",
    "bb_lower_rel",
    "bb_width_rel",
    "ema_20_rel",
    "ema_50_rel",
    "obv_delta",
    "stoch_k",
    "cci_20",
    "day_of_week",
    "month",
    "vix_close",
    "log_ret",
    "high_ratio",
    "low_ratio",
    "close_ratio",
    "vol_delta",
    "vol_annual_pct",
    "news_sentiment",
)

# Dagens-open-feature appendes KUN når OPEN_FEATURE_ENABLED er til (mellem basis og makro),
# så flag fra => uændret feature-sæt. next_open_gap = log-gap fra close_t til open_{t+1}.
_OPEN_FEATURE_COLUMNS: tuple[str, ...] = (
    ("next_open_gap",) if getattr(config, "OPEN_FEATURE_ENABLED", False) else ()
)

# Markeds-brede makro-kolonner (krise-signaler + olie) appendes KUN når flag er til, så
# flag fra => uændret 22-feature-sæt. config ejer navnene; her ejes beregningen.
_MACRO_FEATURE_COLUMNS: tuple[str, ...] = tuple(getattr(config, "MACRO_FEATURE_COLUMNS", ()))

# Autoritativ feature-liste (skal matche config.N_FEATURES). Dynamisk efter flag,
# så engineer_features og scaler/model altid er enige om kolonnesættet.
FEATURE_COLUMNS: tuple[str, ...] = (
    _BASE_FEATURE_COLUMNS
    + _OPEN_FEATURE_COLUMNS
    + (_MACRO_FEATURE_COLUMNS if getattr(config, "MACRO_FEATURES_ENABLED", False) else ())
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


def obv_delta(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Skalainvariant OBV-ændring: fortegn af daglig prisændring gange dagens volumen
    relativt til eget rullende gennemsnit. Erstatter kumulativt OBV (ikke-stationært,
    skalerer med aktiens absolutte volumen) med en retnings-vægtet relativ-volumen-puls.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    vol = volume.astype(float)
    avg_vol = vol.rolling(window=window, min_periods=window).mean().replace(0, np.nan)
    return direction * vol / avg_vol


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
    """Alle 21 feature-kolonner (uden dropna), index = dato. Kræver OHLCV (+ valgfri vix_close).

    Alle pris-afledte features er skalainvariante: indikatorer normaliseres med close
    (eller er allerede afgrænsede), og rå OHLCV erstattes af log-afkast, intradag-ratios
    (H/O, L/O, C/O) og volumen-delta.
    """
    df = df.sort_index().copy()
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # Nævnere der må være nul (fx volumen-stop) → NaN i stedet for inf.
    close_nz = close.replace(0, np.nan)
    open_nz = open_.replace(0, np.nan)
    vol_nz = vol.replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi(close, 14)
    # MACD i pris-enheder normaliseret med close → skalainvariant.
    macd_l, sig = macd_signal(close)
    out["macd_norm"] = macd_l / close_nz
    out["macd_signal_norm"] = sig / close_nz
    out["macd_hist_norm"] = (macd_l - sig) / close_nz
    # Bollinger-bånd som relativ afstand fra close (bredde som fraktion af close).
    up, lo, width = bollinger_bands(close)
    out["bb_upper_rel"] = up / close_nz - 1.0
    out["bb_lower_rel"] = lo / close_nz - 1.0
    out["bb_width_rel"] = width / close_nz
    # EMA som prisens relative afstand fra glidende gennemsnit (+ = over MA).
    out["ema_20_rel"] = close / ema(close, 20).replace(0, np.nan) - 1.0
    out["ema_50_rel"] = close / ema(close, 50).replace(0, np.nan) - 1.0
    out["obv_delta"] = obv_delta(close, vol)
    out["stoch_k"] = stochastic_k(high, low, close, 14)
    out["cci_20"] = cci(high, low, close, 20)

    idx = pd.DatetimeIndex(df.index)
    out["day_of_week"] = idx.dayofweek.astype(float)
    out["month"] = idx.month.astype(float)

    if "vix_close" in df.columns:
        out["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce").ffill()
    else:
        out["vix_close"] = np.nan

    # Stationære pris-/volumen-features (ændring frem for niveau).
    out["log_ret"] = np.log(close / close.shift(1))
    out["high_ratio"] = high / open_nz
    out["low_ratio"] = low / open_nz
    out["close_ratio"] = close / open_nz
    out["vol_delta"] = np.log(vol_nz / vol_nz.shift(1))
    if "vol_annual_pct" in df.columns:
        out["vol_annual_pct"] = pd.to_numeric(df["vol_annual_pct"], errors="coerce")
    else:
        out["vol_annual_pct"] = rolling_annualized_log_vol_pct(close)

    # Nyheds-sentiment (finBERT, p_pos − p_neg ∈ [-1, 1]). Manglende/ukendte dage → 0.0
    # (neutral) i stedet for NaN, så engineer_features ikke dropper rækker uden nyheder.
    if "news_sentiment" in df.columns:
        out["news_sentiment"] = pd.to_numeric(df["news_sentiment"], errors="coerce").fillna(0.0)
    else:
        out["news_sentiment"] = 0.0

    # Dagens-open-gap: log-gap fra dag t's close til dag t+1's open. Live kører lige efter
    # åbning, så næste dags open er kendt ved trade-tid — samme alignment som target
    # (open→close næste dag), derfor ingen leakage. Sidste række er NaN (intet t+1) og
    # droppes af engineer_features (kan ej være træningssample; target er også NaN der).
    # Beregnes altid; FEATURE_COLUMNS afgør om den faktisk indgår (flag-styret).
    out["next_open_gap"] = np.log(open_.shift(-1) / close_nz)

    # Markeds-brede makro-/krise-signaler (samme mønster som vix_close): læs fra base-
    # kolonnen hvis til stede (ffill), ellers neutralværdi — så rækker aldrig droppes.
    # Beregnes altid; FEATURE_COLUMNS afgør om de faktisk indgår (flag-styret).
    neutral = getattr(config, "MACRO_FEATURE_NEUTRAL", {})
    for col in _MACRO_FEATURE_COLUMNS:
        fill = float(neutral.get(col, 0.0))
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(fill)
        else:
            out[col] = fill

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


def build_dataset_frame(
    ohlcv: pd.DataFrame,
    vix_close: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fuldt datasæt til cache: OHLCV + vol_annual_pct + vix_close + (valgfri makro-
    krise-signaler) + alle FEATURE_COLUMNS (uden dropna, så fuld historik bevares
    med NaN-opvarmning).

    ``macro``: markeds-bred frame (date-index) med kolonner i config.MACRO_FEATURE_COLUMNS.
    Joines på base-index (ffill) som vix_close, så de materialiseres i cachen og bæres
    med ved senere inkrementel merge. Manglende kolonner/dage fyldes neutralt i features.
    """
    base = ohlcv.sort_index().copy()
    if vix_close is not None:
        vix = pd.to_numeric(vix_close, errors="coerce")
        base["vix_close"] = vix.reindex(base.index).ffill()
    if macro is not None and not macro.empty:
        m = macro.sort_index()
        m.index = pd.to_datetime(m.index)
        for col in _MACRO_FEATURE_COLUMNS:
            if col in m.columns:
                base[col] = pd.to_numeric(m[col], errors="coerce").reindex(base.index).ffill()
    base["vol_annual_pct"] = rolling_annualized_log_vol_pct(base["close"])
    feats = _feature_frame(base)
    # Undgå dublerede kolonner (vix_close/makro findes både i base og feats).
    extra = feats.drop(columns=[c for c in feats.columns if c in base.columns], errors="ignore")
    return pd.concat([base, extra], axis=1)


def targets_next_day_open_to_close_pct(ohlcv: pd.DataFrame, idx: pd.Index) -> pd.Series:
    """Næste handelsdags intradag-afkast (open→close) i pct, alignet på dag t før hop."""
    open_ = ohlcv["open"].reindex(idx).astype(float)
    close = ohlcv["close"].reindex(idx).astype(float)
    nxt_open = open_.shift(-1)
    nxt_close = close.shift(-1)
    return (nxt_close / nxt_open.replace(0, np.nan) - 1.0) * 100.0
