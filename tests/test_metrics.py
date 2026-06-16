"""Enheds-tests for metrics.py og regime-rapporteringen i backtest.py.

Ingen model eller cache kræves — alt er syntetisk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import metrics  # noqa: E402


# --------------------------- metrics.py ---------------------------

def test_max_drawdown_monotonic_up_is_zero():
    eq = pd.Series([100.0, 101.0, 102.5, 110.0])
    assert metrics.max_drawdown(eq) == pytest.approx(0.0)


def test_max_drawdown_known_dip():
    eq = pd.Series([100.0, 80.0, 90.0])  # 100 -> 80 = -20%
    assert metrics.max_drawdown(eq) == pytest.approx(-20.0)


def test_sharpe_sign_follows_mean():
    pos = pd.Series([0.1, 0.2, 0.05, 0.15])
    assert metrics.sharpe(pos) > 0
    assert metrics.sharpe(-pos) < 0


def test_sharpe_nan_on_zero_variance():
    assert np.isnan(metrics.sharpe(pd.Series([0.5, 0.5, 0.5])))


def test_sortino_inf_when_no_downside():
    assert metrics.sortino(pd.Series([0.1, 0.2, 0.3])) == float("inf")


def test_hit_rate_counts_positive_days():
    assert metrics.hit_rate(pd.Series([1.0, -1.0, 2.0, -3.0])) == pytest.approx(50.0)


def test_turnover_full_rotation_and_static():
    assert metrics.turnover(pd.Series(["A", "B", "C", "D"])) == pytest.approx(100.0)
    assert metrics.turnover(pd.Series(["A", "A", "A", "A"])) == pytest.approx(0.0)


def test_calmar_positive_for_up_with_drawdown():
    eq = pd.Series([100.0, 120.0, 90.0, 150.0])
    cal = metrics.calmar(eq)
    assert np.isfinite(cal) and cal > 0


def test_annualized_return_doubling_in_one_year():
    eq = pd.Series(np.linspace(100.0, 200.0, 253))  # ~1 handelsår, x2
    assert metrics.annualized_return(eq) == pytest.approx(100.0, abs=2.0)


def test_summarize_keys_present():
    eq = pd.Series(np.linspace(100.0, 130.0, 60))
    out = metrics.summarize(eq, symbols=pd.Series(["A"] * 60))
    for k in ("sharpe", "sortino", "calmar", "max_drawdown_pct", "hit_rate_pct", "turnover_pct"):
        assert k in out


# --------------------------- regime report ---------------------------

def _fake_equity(dates: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, len(dates))
    return pd.Series((1 + rets).cumprod() * 100_000.0, index=dates)


def test_regime_report_slices_to_window_dates():
    from stock_predictor.backtest import _regime_report

    dates = pd.bdate_range("2019-06-01", "2020-12-31")
    eq = _fake_equity(dates)
    daily_log = pd.DataFrame({
        "trade_date": [d.date().isoformat() for d in dates],
        "best_symbol": ["AAPL"] * len(dates),
    })
    records = _regime_report({"best": eq, "avg": eq}, daily_log, 2019, 2020)

    covid = [r for r in records if r["window"] == "COVID crash" and r["strategy"] == "best"]
    assert covid, "COVID crash regime should be present"
    r = covid[0]
    # COVID-vinduet er 2020-02-19..2020-03-23 — slicet må holde sig indenfor.
    assert r["start"] >= "2020-02-19" and r["end"] <= "2020-03-23"
    assert r["n_days"] >= 5
    # Pr-år-vinduer findes også.
    assert any(rr["window"] == "2019" and rr["window_type"] == "year" for rr in records)


def test_regime_report_skips_out_of_range_regimes():
    from stock_predictor.backtest import _regime_report

    dates = pd.bdate_range("2023-01-01", "2024-12-31")
    eq = _fake_equity(dates)
    daily_log = pd.DataFrame({
        "trade_date": [d.date().isoformat() for d in dates],
        "best_symbol": ["MSFT"] * len(dates),
    })
    records = _regime_report({"best": eq}, daily_log, 2023, 2024)
    # GFC 2008 ligger uden for intervallet → ingen GFC-record.
    assert not any(r["window"] == "GFC 2008" for r in records)


def test_regime_report_overall_row_equals_full_return():
    from stock_predictor.backtest import _regime_report

    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    eq = _fake_equity(dates)
    daily_log = pd.DataFrame({
        "trade_date": [d.date().isoformat() for d in dates],
        "best_symbol": ["AAPL"] * len(dates),
    })
    records = _regime_report({"best": eq}, daily_log, 2019, 2020)
    overall = [r for r in records if r["window_type"] == "overall" and r["strategy"] == "best"]
    assert len(overall) == 1
    expected = (float(eq.iloc[-1]) / float(eq.iloc[0]) - 1.0) * 100.0
    assert overall[0]["total_return_pct"] == pytest.approx(round(expected, 2), abs=0.02)
    assert overall[0]["n_days"] == len(eq)


# --------------------------- VIX risk-off gate ---------------------------

def test_vix_gate_zeroes_risk_off_days():
    from stock_predictor.backtest import _vix_gated_equities, START_EQUITY

    dates = pd.bdate_range("2020-01-01", periods=6)
    # best_actual: +2% hver dag; to dage skal nulles af VIX-filteret.
    daily_log = pd.DataFrame({
        "trade_date": [d.date().isoformat() for d in dates],
        "best_actual": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "avg_actual": [1.0] * 6,
    })
    # vix_decision (forrige dags VIX): dag 3 og 5 er ≥ 30 → kontant.
    vix_decision = pd.Series([15, 15, 35, 20, 40, 18], index=dates, dtype=float)

    gated, n_cash, n_total = _vix_gated_equities(daily_log, vix_decision, threshold=30.0)
    assert n_cash == 2 and n_total == 6
    eq = gated["best"].to_numpy()
    # 4 handelsdage à +2% (de to risk-off-dage giver 0 %): 1.02**4.
    assert eq[-1] == pytest.approx(START_EQUITY * 1.02 ** 4, rel=1e-9)
    # Equity står stille hen over en risk-off-dag (index 2: dag 3).
    assert eq[2] == pytest.approx(eq[1], rel=1e-12)


def test_vix_gate_no_vix_means_no_cash():
    from stock_predictor.backtest import _vix_gated_equities, START_EQUITY

    dates = pd.bdate_range("2021-01-01", periods=4)
    daily_log = pd.DataFrame({
        "trade_date": [d.date().isoformat() for d in dates],
        "best_actual": [1.0, 1.0, 1.0, 1.0],
        "avg_actual": [1.0, 1.0, 1.0, 1.0],
    })
    gated, n_cash, n_total = _vix_gated_equities(daily_log, pd.Series(dtype=float), threshold=30.0)
    assert n_cash == 0 and n_total == 4
    assert gated["best"].to_numpy()[-1] == pytest.approx(START_EQUITY * 1.01 ** 4, rel=1e-9)
