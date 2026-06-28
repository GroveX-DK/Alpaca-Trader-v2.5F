"""Enheds-tests for CPCV-fold-logikken i cpcv.py (purge/embargo, grupper, sti-geometri).

Ingen model, GPU eller cache kræves — alt er syntetisk og hurtigt (ingen genoptræning).
Det vigtigste her er purge: ingen trænings-vindue må lække ind i en test-blok.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import cpcv  # noqa: E402
from stock_predictor.train import SymbolData, WindowRec  # noqa: E402


# --------------------------- _assign_groups ---------------------------

def test_assign_groups_equal_contiguous():
    dates = pd.date_range("2020-01-01", periods=30, freq="B").to_numpy()
    d2g, bounds = cpcv._assign_groups(dates, 6)
    assert len(bounds) == 6
    # 30 dage / 6 grupper = 5 hver, sammenhængende og ikke-overlappende.
    counts: dict[int, int] = {}
    for g in d2g.values():
        counts[g] = counts.get(g, 0) + 1
    assert counts == {g: 5 for g in range(6)}
    for g in range(1, 6):
        assert bounds[g - 1][1] < bounds[g][0]


def test_assign_groups_too_few_days_raises():
    dates = pd.date_range("2020-01-01", periods=3, freq="B").to_numpy()
    with pytest.raises(ValueError):
        cpcv._assign_groups(dates, 6)


# --------------------------- purge/embargo ---------------------------

def _single_symbol(n_rows: int, seq: int):
    all_dates = pd.date_range("2019-06-01", periods=n_rows, freq="B")
    mat = np.zeros((n_rows, 4), dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.float32)
    data = {"AAA": SymbolData(matrix=mat, y=y, end_dates=all_dates.to_numpy())}
    recs = [
        WindowRec(sym="AAA", end_i=i, y=0.0, end_date=all_dates[i].to_numpy())
        for i in range(seq - 1, n_rows - 1)
    ]
    return data, recs, all_dates


def test_purge_removes_all_leakage():
    """Ethvert overlevende trænings-vindue må IKKE overlappe test-blokken (+embargo)."""
    seq = 5
    data, recs, _ = _single_symbol(60, seq)
    period = np.array([r.end_date for r in recs if pd.Timestamp(r.end_date) >= pd.Timestamp("2019-08-01")],
                      dtype="datetime64[ns]")
    d2g, bounds = cpcv._assign_groups(period, 4)
    meta = cpcv._rec_meta(recs, data, d2g, seq)
    embargo = pd.Timedelta(days=10)
    test_groups = {1}
    gs, ge = bounds[1]
    train = cpcv._purge_train(meta, {0, 2, 3}, [bounds[1]], embargo)
    assert train, "purge fjernede alt — testen er meningsløs"
    for r in train:
        ws = pd.Timestamp(data[r.sym].end_dates[r.end_i - seq + 1])
        tg = pd.Timestamp(data[r.sym].end_dates[r.end_i + 1])
        # ingen overlap af [ws, tg] med [gs, ge+embargo]
        assert not (ws <= ge + embargo and tg >= gs)


def test_purge_only_keeps_train_groups():
    seq = 5
    data, recs, _ = _single_symbol(60, seq)
    period = np.array([r.end_date for r in recs if pd.Timestamp(r.end_date) >= pd.Timestamp("2019-08-01")],
                      dtype="datetime64[ns]")
    d2g, bounds = cpcv._assign_groups(period, 4)
    meta = cpcv._rec_meta(recs, data, d2g, seq)
    train = cpcv._purge_train(meta, {0, 3}, [bounds[1], bounds[2]], pd.Timedelta(days=0))
    # alle overlevende skal tilhøre en trænings-gruppe
    g_by_rec = {id(m["rec"]): m["group"] for m in meta}
    assert all(g_by_rec[id(r)] in {0, 3} for r in train)


# --------------------------- sti-geometri ---------------------------

@pytest.mark.parametrize("n,k,exp_combos,exp_phi", [(6, 2, 15, 5), (5, 2, 10, 4), (8, 2, 28, 7)])
def test_path_geometry(n, k, exp_combos, exp_phi):
    combos = list(combinations(range(n), k))
    cwg = {g: [ci for ci, tg in enumerate(combos) if g in tg] for g in range(n)}
    phi = len(cwg[0])
    assert len(combos) == exp_combos
    assert phi == exp_phi == exp_combos * k // n
    assert all(len(cwg[g]) == phi for g in range(n))


# --------------------------- simulering på OOS-maps ---------------------------

def test_simulate_metrics_from_maps():
    dates = pd.date_range("2021-01-01", periods=20, freq="B")
    rng = np.random.default_rng(0)
    pmap, amap = {}, {}
    for ts in dates:
        for s in ["AAA", "BBB", "CCC"]:
            pmap[(ts, s)] = float(rng.standard_normal())
            amap[(ts, s)] = float(rng.standard_normal())
    m_best, m_avg, eq = cpcv._simulate_metrics(pmap, amap, min_syms=2)
    assert eq is not None and len(eq) > 0
    assert "sharpe" in m_best and "total_return_pct" in m_best


def test_simulate_metrics_min_syms_filters_all():
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    pmap = {(ts, "AAA"): 1.0 for ts in dates}
    amap = {(ts, "AAA"): 0.5 for ts in dates}
    # kun 1 symbol/dag, men min_syms=2 → ingen handelsdage tilbage
    m_best, m_avg, eq = cpcv._simulate_metrics(pmap, amap, min_syms=2)
    assert eq is None and m_best == {}


# --------------------------- auto-detect + feasibility guard ---------------------------

def _recs_for_dates(dates: pd.DatetimeIndex, n_syms: int) -> list:
    """n_syms symboler med et gyldigt vindue på hver dato (end_i er irrelevant her)."""
    out = []
    for d in dates:
        for s in range(n_syms):
            out.append(WindowRec(sym=f"S{s}", end_i=0, y=0.0, end_date=d.to_numpy()))
    return out


def test_auto_start_year_first_qualifying_year():
    # Kun 3 symboler i 2020, 12 fra 2021-02 → med min_syms=10 vælges 2021.
    d2020 = pd.date_range("2020-03-01", periods=50, freq="B")
    d2021 = pd.date_range("2021-02-01", periods=200, freq="B")
    recs = _recs_for_dates(d2020, 3) + _recs_for_dates(d2021, 12)
    assert cpcv._auto_start_year(recs, end_year=2026, min_syms=10) == 2021


def test_auto_start_year_bumps_when_second_half():
    # Første kvalificerende dato i august → ryk til næste år for fyldigere første gruppe.
    d = pd.date_range("2020-08-03", periods=100, freq="B")
    recs = _recs_for_dates(d, 12)
    assert cpcv._auto_start_year(recs, end_year=2026, min_syms=10) == 2021


def test_auto_start_year_raises_when_too_thin():
    d = pd.date_range("2024-01-01", periods=20, freq="B")
    recs = _recs_for_dates(d, 3)  # aldrig ≥10 symboler
    with pytest.raises(RuntimeError):
        cpcv._auto_start_year(recs, end_year=2026, min_syms=10)


def test_check_feasible_raises_on_thin_groups():
    # 24 handelsdage / 6 grupper = 4 dage/gruppe « _MIN_GROUP_TRADING_DAYS → afbryd.
    seq = 5
    data, recs, _ = _single_symbol(60, seq)
    period = np.array([r.end_date for r in recs][:24], dtype="datetime64[ns]")
    d2g, bounds = cpcv._assign_groups(period, 6)
    meta = cpcv._rec_meta(recs, data, d2g, seq)
    with pytest.raises(RuntimeError):
        cpcv._check_feasible(meta, bounds, 6, 2, pd.Timedelta(days=0), start_year=2019)


def test_check_feasible_raises_when_history_below_seqlen(monkeypatch):
    """Grupper er store nok (>=40 dage), men total label-historik < SEQ_LEN → afbryd (delt kontekst)."""
    seq = 5
    monkeypatch.setattr(cpcv.config, "SEQ_LEN", 1000)  # lookback >> de 240 labelede dage
    data, recs, _ = _single_symbol(300, seq)
    period = np.array([r.end_date for r in recs][:240], dtype="datetime64[ns]")  # 40/gruppe
    d2g, bounds = cpcv._assign_groups(period, 6)
    meta = cpcv._rec_meta(recs, data, d2g, seq)
    with pytest.raises(RuntimeError, match="SEQ_LEN"):
        cpcv._check_feasible(meta, bounds, 6, 2, pd.Timedelta(days=0), start_year=2024)
