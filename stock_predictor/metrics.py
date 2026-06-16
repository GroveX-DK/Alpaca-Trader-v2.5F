"""Risiko-/performance-nøgletal til backtesten (måleinstrumentet).

Rene funktioner uden I/O: de tager en equity-kurve og/eller daglige afkast og returnerer
skalarer. Daglige afkast forventes i **procent** (samme enhed som ``daily_log``'s actual-
kolonner og ``equity.pct_change()*100``). Annualisering bruger 252 handelsdage.

Hvorfor: et enkelt års samlede afkast er luk-domineret og siger intet om risiko. Sharpe,
Sortino, Calmar, max drawdown, hit rate og turnover gør performance sammenlignelig på tværs
af markedsregimer (se regime-backtesten i backtest.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _as_fraction(daily_ret_pct: pd.Series) -> np.ndarray:
    """Daglige afkast i pct → fraktioner, NaN droppet."""
    arr = pd.to_numeric(daily_ret_pct, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr / 100.0


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Daglige afkast i **procent** udledt af en equity-kurve (første dag droppes)."""
    if equity is None or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.astype(float).pct_change().dropna() * 100.0


def max_drawdown(equity: pd.Series) -> float:
    """Værste top-til-bund-fald på equity-kurven i pct (negativ tal, fx -42.0)."""
    if equity is None or len(equity) == 0:
        return float("nan")
    vals = equity.astype(float).to_numpy()
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    return float(dd.min()) * 100.0


def annualized_return(equity: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometrisk annualiseret afkast i pct ud fra equity-kurvens start/slut og længde."""
    if equity is None or len(equity) < 2:
        return float("nan")
    vals = equity.astype(float).to_numpy()
    total = vals[-1] / vals[0]
    n = len(vals) - 1
    if total <= 0 or n <= 0:
        return float("nan")
    return (total ** (periods / n) - 1.0) * 100.0


def annualized_vol(daily_ret_pct: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualiseret std af daglige afkast i pct."""
    r = _as_fraction(daily_ret_pct)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1)) * np.sqrt(periods) * 100.0


def sharpe(daily_ret_pct: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualiseret Sharpe (rf=0): mean/std × √periods. NaN hvis std≈0."""
    r = _as_fraction(daily_ret_pct)
    if r.size < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(r) / sd) * np.sqrt(periods)


def sortino(daily_ret_pct: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualiseret Sortino: mean / downside-std (kun negative afkast). inf hvis intet downside."""
    r = _as_fraction(daily_ret_pct)
    if r.size < 2:
        return float("nan")
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf")
    dd = np.sqrt(np.mean(np.square(downside)))
    if dd < 1e-12:
        return float("nan")
    return float(np.mean(r) / dd) * np.sqrt(periods)


def calmar(equity: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualiseret afkast / |max drawdown|. NaN hvis drawdown≈0."""
    mdd = max_drawdown(equity)
    if not np.isfinite(mdd) or abs(mdd) < 1e-9:
        return float("nan")
    ann = annualized_return(equity, periods)
    if not np.isfinite(ann):
        return float("nan")
    return ann / abs(mdd)


def hit_rate(daily_ret_pct: pd.Series) -> float:
    """Andel handelsdage med positivt afkast i pct (0–100)."""
    r = _as_fraction(daily_ret_pct)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0)) * 100.0


def turnover(symbols: pd.Series | list) -> float:
    """Andel dage hvor pick'et skiftede ift. dagen før (0–100).

    Fuld daglig rotation (forskelligt symbol hver dag) ≈ 100 %, dvs. ~200 % notional
    (sælg gammelt + køb nyt). Tomme/NaN-symboler springes over.
    """
    syms = [s for s in (symbols.tolist() if isinstance(symbols, pd.Series) else list(symbols))
            if isinstance(s, str) and s]
    if len(syms) < 2:
        return float("nan")
    changes = sum(1 for a, b in zip(syms[:-1], syms[1:]) if a != b)
    return changes / (len(syms) - 1) * 100.0


def summarize(
    equity: pd.Series,
    daily_ret_pct: pd.Series | None = None,
    symbols: pd.Series | list | None = None,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Saml alle nøgletal til ét dict (afrundet) til JSON-sidecar/visning."""
    if daily_ret_pct is None:
        daily_ret_pct = daily_returns_from_equity(equity)
    out = {
        "total_return_pct": round((float(equity.iloc[-1]) / float(equity.iloc[0]) - 1.0) * 100.0, 2)
        if equity is not None and len(equity) >= 1 else float("nan"),
        "annualized_return_pct": round(annualized_return(equity, periods), 2),
        "annualized_vol_pct": round(annualized_vol(daily_ret_pct, periods), 2),
        "sharpe": round(sharpe(daily_ret_pct, periods), 3),
        "sortino": round(sortino(daily_ret_pct, periods), 3),
        "calmar": round(calmar(equity, periods), 3),
        "max_drawdown_pct": round(max_drawdown(equity), 2),
        "hit_rate_pct": round(hit_rate(daily_ret_pct), 2),
    }
    if symbols is not None:
        out["turnover_pct"] = round(turnover(symbols), 2)
    return out
