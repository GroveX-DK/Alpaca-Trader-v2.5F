"""Combinatorial Purged Cross-Validation (CPCV) backtest — modellens *holdbarhed*.

Hvor regime-backtesten (``backtest.run_regime_backtest``) kører ÉN færdigtrænet model over
historikken (inferens, ingen genoptræning), måler CPCV noget andet: hvor robust selve
strategien er over for *hvilke* perioder modellen trænes/testes på. Det er det rigtige værktøj
til overfitting/holdbarhed (López de Prado, "Advances in Financial Machine Learning", kap. 7+12).

Metode (N grupper, k test-grupper pr. kombination):
  1. Den labelede periode [start_year, end_year] deles i ``N`` sammenhængende tids-grupper.
  2. For hver af C(N, k) kombinationer bruges k grupper som *test* og resten som *træning*.
  3. Trænings-vinduer **purges** (fjernes) hvis deres feature-vindue (SEQ_LEN tilbage) eller
     deres label (næste dag) overlapper en test-gruppe — plus en **embargo** efter test-blokken.
     Dette dræber både fremad-leakage (label rækker ind i test) og bagud-leakage (et trænings-
     vindue efter test ser test-data i sit lookback). Med SEQ_LEN≈1000 er purge stor.
  4. En frisk model trænes pr. kombination (genbruger ``train._fit_lstm``) og scorer test-grupperne.
  5. Test-forudsigelserne samles til φ = C(N,k)·k/N *backtest-stier* (hver gruppe testes i
     C(N-1,k-1) kombinationer). Hver sti giver én OOS equity-kurve → en *fordeling* af
     Sharpe/afkast/MaxDD = holdbarhedsmålet.

DYRT: k modeller × C(N,k) genoptræninger. Med N=6,k=2 = 15 træninger. Kør på GPU-desktop
(se ``config.TRAIN_DEVICE``), ikke på laptop — på CPU er det dage.

Kør:
    python -m stock_predictor.main --cpcv-backtest 2010
    python -m stock_predictor.cpcv

Bemærk: med én fast model-arkitektur er klassisk PBO (over et strategi-grid) ikke direkte
relevant; holdbarheden aflæses i fordelingen af OOS-Sharpe på tværs af stierne + andelen af
stier med Sharpe ≤ 0.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import config  # noqa: E402
from stock_predictor import metrics as _metrics  # noqa: E402
from stock_predictor.backtest import START_EQUITY, _BACKTEST_DIR, _simulate  # noqa: E402
from stock_predictor.data_fetcher import fetch_daily_bars  # noqa: E402
from stock_predictor.torch_device import resolve_device  # noqa: E402
from stock_predictor.train import (  # noqa: E402
    SymbolData,
    WindowRec,
    _build_index,
    _engineer_all,
    _fit_lstm,
    _fit_scaler,
    _scale_all,
    _subsample_train_stride,
    _time_sort_split,
)

logger = logging.getLogger(__name__)

# Antal sekvenser pr. model-forward ved test-scoring (CPU/RAM-venligt).
INFER_BATCH = 256

# Holdbarheds-guard: en CPCV-kørsel er kun meningsfuld hvis hver gruppe har nok handelsdage
# OG den hårdest purgede kombination beholder nok trænings-vinduer. Med SEQ_LEN≈1000 og
# kort historik bliver folds ellers tomme (jf. data-begrænsningen i cachen).
_MIN_GROUP_TRADING_DAYS = 40
_MIN_TRAIN_WINDOWS = 200

_CPCV_LIMITATIONS = [
    "Survivorship bias: watchlisten er nutidens vindere; afnoterede tabere mangler.",
    "news_sentiment er 0 (inert) før ~2015 — let input-skift i de ældre grupper.",
    "Med SEQ_LEN≈1000 purges store dele af træningssættet nær hver test-gruppe — "
    "fold-modellerne ser færre vinduer end fuld-træning; tidlige grupper er tyndest.",
    "Én fast arkitektur: dette måler holdbarhed (fordeling af OOS-performance), ikke "
    "klassisk PBO over et strategi-grid.",
    "Ingen handelsomkostninger/slippage; all-in top-1 (best) og ligevægt top-3 (avg).",
]


# ------------------------------------------------------------------ fold-geometri

def _assign_groups(end_dates: np.ndarray, n_groups: int) -> Tuple[dict, list]:
    """Del de unikke handelsdage i ``n_groups`` sammenhængende grupper (≈ lige mange dage).

    Returnerer ``(date_to_group, group_bounds)`` hvor ``date_to_group`` mapper en
    ``pd.Timestamp`` → gruppe-id, og ``group_bounds[g] = (start_ts, end_ts)``.
    """
    uniq = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(end_dates))))
    if len(uniq) < n_groups:
        raise ValueError(f"For få handelsdage ({len(uniq)}) til {n_groups} grupper.")
    chunks = np.array_split(np.arange(len(uniq)), n_groups)
    date_to_group: dict[pd.Timestamp, int] = {}
    group_bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for g, idx in enumerate(chunks):
        block = uniq[idx]
        group_bounds.append((block[0], block[-1]))
        for ts in block:
            date_to_group[ts] = g
    return date_to_group, group_bounds


def _rec_meta(
    recs: List[WindowRec],
    data_by_sym: dict[str, SymbolData],
    date_to_group: dict,
    seq_len: int,
) -> list[dict]:
    """Forbered pr.-vindue metadata til purge/gruppering (beregnes én gang).

    For hvert vindue: gruppe (via slutdato), feature-vinduets startdato (SEQ_LEN tilbage) og
    label-datoen (næste handelsdag). Vinduer hvis slutdato falder uden for den labelede periode
    (ikke i ``date_to_group``) udelades.
    """
    meta: list[dict] = []
    for r in recs:
        ts_end = pd.Timestamp(r.end_date)
        g = date_to_group.get(ts_end)
        if g is None:
            continue
        d = data_by_sym[r.sym]
        win_start = pd.Timestamp(d.end_dates[r.end_i - seq_len + 1])
        nxt = r.end_i + 1
        target = pd.Timestamp(d.end_dates[nxt]) if nxt < len(d.end_dates) else ts_end + pd.Timedelta(days=1)
        meta.append({"rec": r, "group": g, "win_start": win_start, "target": target, "end": ts_end})
    return meta


def _purge_train(
    meta: list[dict],
    train_groups: set[int],
    test_blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
    embargo: pd.Timedelta,
) -> List[WindowRec]:
    """Vælg trænings-vinduer i ``train_groups`` og purge dem der lækker mod en test-blok.

    Et vindue lækker hvis dets info-interval ``[win_start, target]`` overlapper en test-blok
    ``[g_start, g_end]`` (g_end udvidet med ``embargo``). Det fanger både label-overlap og det
    lange SEQ_LEN-lookback der ellers ville se test-data.
    """
    out: List[WindowRec] = []
    for m in meta:
        if m["group"] not in train_groups:
            continue
        leaks = False
        for (gs, ge) in test_blocks:
            if m["win_start"] <= ge + embargo and m["target"] >= gs:
                leaks = True
                break
        if not leaks:
            out.append(m["rec"])
    return out


def _auto_start_year(recs: List[WindowRec], end_year: int, min_syms: int) -> int:
    """Find det tidligste år hvor ≥ ``min_syms`` symboler har et gyldigt vindue samme dag.

    ``recs`` indeholder kun gyldige vinduer (nok lookback + endeligt target), så den første
    dato med nok navne markerer hvornår universet reelt kan forudsiges. Ligger den dato i
    årets anden halvdel, rykkes der til næste år for en fyldigere første gruppe.
    """
    by_date: dict[pd.Timestamp, int] = {}
    for r in recs:
        ts = pd.Timestamp(r.end_date)
        by_date[ts] = by_date.get(ts, 0) + 1
    qualifying = sorted(ts for ts, c in by_date.items() if c >= min_syms)
    if not qualifying:
        raise RuntimeError(
            f"Ingen handelsdag har ≥{min_syms} symboler med gyldigt vindue — "
            "cachen er for tynd. Kør tools.gather_history_yfinance for fuld historik."
        )
    first = qualifying[0]
    yr = first.year + 1 if first.month > 6 else first.year
    return min(yr, end_year)


def _check_feasible(
    meta: list[dict],
    group_bounds: list[tuple[pd.Timestamp, pd.Timestamp]],
    n_groups: int,
    k_test: int,
    embargo: pd.Timedelta,
    start_year: int,
) -> None:
    """Afbryd tidligt med en handlingsanvisende fejl hvis folds bliver for tynde.

    Tjekker (a) at den DISTINKTE labelede historik er ≥ SEQ_LEN (ellers deler alle vinduer
    stort set samme lookback-kontekst → CPCV bliver meningsløs), (b) at hver gruppe har nok
    handelsdage, og (c) at den hårdest purgede kombination (de ``k`` midterste grupper som test)
    beholder ≥ ``_MIN_TRAIN_WINDOWS`` trænings-vinduer.
    """
    per_group_days = [0] * n_groups
    seen: list[set] = [set() for _ in range(n_groups)]
    for m in meta:
        g = m["group"]
        if m["end"] not in seen[g]:
            seen[g].add(m["end"])
            per_group_days[g] += 1
    total_days = sum(per_group_days)
    seq_len = int(config.SEQ_LEN)
    hint = (
        "Backfill fuld historik (python -m stock_predictor.tools.gather_history_yfinance) "
        "eller sænk SEQ_LEN for CPCV-studiet."
    )
    # (a) Distinkt label-historik skal mindst dække lookbacket, ellers deler træ-/test-vinduer
    #     næsten identisk 4-årig kontekst (purge fanger det ikke — det handler om delt historik).
    if total_days < seq_len:
        raise RuntimeError(
            f"Kun {total_days} distinkte labelede handelsdage fra {start_year} < SEQ_LEN={seq_len}: "
            f"alle vinduer deler stort set samme lookback → CPCV er ikke meningsfuld. {hint}"
        )
    thin = [g for g, d in enumerate(per_group_days) if d < _MIN_GROUP_TRADING_DAYS]
    if thin:
        raise RuntimeError(
            f"For tynde grupper {thin} (<{_MIN_GROUP_TRADING_DAYS} handelsdage) fra {start_year}: "
            f"dage/gruppe={per_group_days}. {hint}"
        )
    # Værst purgede kombination: de k midterste grupper.
    mid = n_groups // 2
    test_groups = list(range(max(0, mid - k_test // 2), max(0, mid - k_test // 2) + k_test))
    test_blocks = [group_bounds[g] for g in test_groups]
    train_groups = set(range(n_groups)) - set(test_groups)
    survivors = len(_purge_train(meta, train_groups, test_blocks, embargo))
    if survivors < _MIN_TRAIN_WINDOWS:
        raise RuntimeError(
            f"Efter purge beholder den hårdeste kombination kun {survivors} trænings-vinduer "
            f"(<{_MIN_TRAIN_WINDOWS}) — SEQ_LEN={int(config.SEQ_LEN)} purger for meget af de "
            f"{start_year}+-data. {hint}"
        )


# ------------------------------------------------------------------ scoring + simulering

def _predict_recs(
    model,
    scaled_by_sym: dict[str, np.ndarray],
    recs: List[WindowRec],
    device: torch.device,
    seq_len: int,
    n_features: int,
) -> Dict[Tuple[pd.Timestamp, str], float]:
    """Batch-scoring: returnér ``{(slutdato, symbol): forudsagt open→close %}`` for ``recs``."""
    preds: Dict[Tuple[pd.Timestamp, str], float] = {}
    model.eval()
    windows: list[np.ndarray] = []
    keys: list[tuple[pd.Timestamp, str]] = []
    with torch.no_grad():
        for r in recs:
            window = scaled_by_sym[r.sym][r.end_i - seq_len + 1 : r.end_i + 1]
            windows.append(np.ascontiguousarray(window, dtype=np.float32))
            keys.append((pd.Timestamp(r.end_date), r.sym))
            if len(windows) >= INFER_BATCH:
                _flush_batch(model, windows, keys, preds, device, seq_len, n_features)
                windows, keys = [], []
        if windows:
            _flush_batch(model, windows, keys, preds, device, seq_len, n_features)
    return preds


def _flush_batch(model, windows, keys, preds, device, seq_len, n_features) -> None:
    xt = torch.from_numpy(np.stack(windows).reshape(len(windows), seq_len, n_features)).to(device)
    out = model(xt).detach().cpu().numpy().reshape(-1)
    for k, v in zip(keys, out):
        preds[k] = float(v)


def _frames_from_maps(
    pred_map: Dict[Tuple[pd.Timestamp, str], float],
    actual_map: Dict[Tuple[pd.Timestamp, str], float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Byg (pred_df, actual_df) indekseret på dato × symbol fra (dato, symbol)-maps."""
    if not pred_map:
        return pd.DataFrame(), pd.DataFrame()
    p_rows = [(ts, sym, v) for (ts, sym), v in pred_map.items()]
    pred_df = (
        pd.DataFrame(p_rows, columns=["date", "sym", "pred"])
        .pivot(index="date", columns="sym", values="pred")
        .sort_index()
    )
    a_rows = [(ts, sym, v) for (ts, sym), v in actual_map.items()]
    actual_df = (
        pd.DataFrame(a_rows, columns=["date", "sym", "act"])
        .pivot(index="date", columns="sym", values="act")
        .reindex(index=pred_df.index, columns=pred_df.columns)
    )
    return pred_df, actual_df


def _simulate_metrics(
    pred_map: Dict[Tuple[pd.Timestamp, str], float],
    actual_map: Dict[Tuple[pd.Timestamp, str], float],
    min_syms: int,
) -> Tuple[dict, dict, pd.Series | None]:
    """Kør dag-for-dag-simuleringen på OOS-forudsigelserne → (metrics_best, metrics_avg, equity_best)."""
    pred_df, actual_df = _frames_from_maps(pred_map, actual_map)
    if pred_df.empty:
        return {}, {}, None
    keep = pred_df.notna().sum(axis=1) >= min_syms
    pred_df, actual_df = pred_df.loc[keep], actual_df.loc[keep]
    if pred_df.empty:
        return {}, {}, None
    _daily, equities = _simulate(pred_df, actual_df)
    eq_best = equities.get("best")
    eq_avg = equities.get("avg")
    m_best = _metrics.summarize(eq_best) if eq_best is not None and len(eq_best) >= 2 else {}
    m_avg = _metrics.summarize(eq_avg) if eq_avg is not None and len(eq_avg) >= 2 else {}
    return m_best, m_avg, eq_best


def _dist(values: list[float]) -> dict:
    """Fordelings-resumé (mean/std/min/median/p25/p75/max) for en liste skalarer."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
        "min": round(float(arr.min()), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.median(arr)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "max": round(float(arr.max()), 3),
    }


# ------------------------------------------------------------------ hoved-entry

def run_cpcv_backtest(
    start_year: int | None = None,
    end_year: int = 2025,
    *,
    n_groups: int | None = None,
    k_test: int | None = None,
    embargo_days: int | None = None,
    show_plot: bool = True,
    output_path: Path | None = None,
) -> dict:
    """Kør CPCV: genoptræn modellen pr. fold-kombination og rapportér OOS-holdbarhed.

    ``start_year=None`` → auto-detektér det tidligste år hvor universet kan forudsiges (efter
    lookback) og som giver folds der overlever purge; ellers afbrydes med en handlingsanvisende
    fejl. ``n_groups``/``k_test``/``embargo_days`` defaulter til config (CPCV_*), typisk 6/2/10.
    Returnerer et dict med per-kombination + per-sti metrikker, fordelings-resumé og stier.
    """
    n_groups = int(n_groups if n_groups is not None else getattr(config, "CPCV_N_GROUPS", 6))
    k_test = int(k_test if k_test is not None else getattr(config, "CPCV_K_TEST", 2))
    embargo_days = int(embargo_days if embargo_days is not None else getattr(config, "CPCV_EMBARGO_DAYS", 10))
    if not (1 <= k_test < n_groups):
        raise ValueError("k_test skal være ≥1 og < n_groups.")

    seq_len = int(config.SEQ_LEN)
    n_features = int(config.N_FEATURES)
    min_syms = int(getattr(config, "MIN_SYMBOLS_PER_DAY", 10))
    embargo = pd.Timedelta(days=embargo_days)
    device = resolve_device(str(getattr(config, "TRAIN_DEVICE", "auto")))

    n_combos = len(list(combinations(range(n_groups), k_test)))
    n_paths = n_combos * k_test // n_groups
    if device.type != "cuda":
        logger.warning(
            "CPCV genoptræner %s modeller — på %s (ikke CUDA) tager det meget lang tid. "
            "Kør hellere på GPU-desktop.", n_combos, device,
        )

    # --- Hent data (cache-first). Ved auto-start hentes al historik (floor 1990) så start-året
    #     kan detekteres; ellers nok warmup til at dække SEQ_LEN-lookback før start_year. ---
    fetch_from_year = 1990 if start_year is None else start_year
    lookback = (end_year - fetch_from_year + 1) * 366 + config.INFERENCE_FETCH_CALENDAR_DAYS
    fetch_result = fetch_daily_bars(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.WATCHLIST,
        end=date(end_year, 12, 31),
        lookback_calendar_days=lookback,
        extra_buffer_days=0,
        prefer_cache_only=True,
    )
    bars = fetch_result.bars
    if not bars:
        raise RuntimeError("Ingen barer fra cache — kør import/--train først.")

    data_by_sym = _engineer_all(bars, seq_len)
    if not data_by_sym:
        raise RuntimeError("Ingen symboler med nok historik. Tjek cache-dækning.")
    recs_all = _build_index(data_by_sym, seq_len)

    # --- Auto-detektér start-år hvis ikke angivet ---
    if start_year is None:
        start_year = _auto_start_year(recs_all, end_year, min_syms)
        logger.info(
            "Auto-detekteret start_year=%s (tidligste år med ≥%s symboler efter lookback).",
            start_year, min_syms,
        )
    if end_year < start_year:
        raise ValueError("end_year skal være ≥ start_year.")
    logger.info(
        "CPCV %s-%s: N=%s grupper, k=%s test → %s genoptræninger, φ=%s stier (device=%s, seq_len=%s).",
        start_year, end_year, n_groups, k_test, n_combos, n_paths, device, seq_len,
    )

    # --- Grupper kun den labelede periode [start_year, end_year] ---
    period_dates = np.array(
        [r.end_date for r in recs_all if start_year <= pd.Timestamp(r.end_date).year <= end_year],
        dtype="datetime64[ns]",
    )
    if period_dates.size == 0:
        raise RuntimeError("Ingen vinduer i [start_year, end_year] — tjek cache/årstal.")
    date_to_group, group_bounds = _assign_groups(period_dates, n_groups)
    meta = _rec_meta(recs_all, data_by_sym, date_to_group, seq_len)

    # Holdbarheds-guard: afbryd med en handlingsanvisende fejl hvis folds er for tynde
    # (typisk for kort historik ift. SEQ_LEN). Forhindrer meningsløse/degenererede kørsler.
    _check_feasible(meta, group_bounds, n_groups, k_test, embargo, start_year)

    logger.info(
        "Grupper: %s | vinduer i perioden: %s | gruppe-grænser: %s",
        n_groups, len(meta),
        ", ".join(f"{g}:{s.date()}..{e.date()}" for g, (s, e) in enumerate(group_bounds)),
    )

    # Aktuelle (target/realiseret) afkast pr. (dato, symbol) — facit til simuleringen.
    actual_all: Dict[Tuple[pd.Timestamp, str], float] = {
        (m["end"], m["rec"].sym): float(m["rec"].y) for m in meta
    }

    combos = list(combinations(range(n_groups), k_test))
    stride = int(getattr(config, "TRAIN_WINDOW_STRIDE", 1))

    # --- Genoptræn + scor pr. kombination ---
    combo_preds: list[Dict[Tuple[pd.Timestamp, str], float]] = []
    combo_rows: list[dict] = []
    for ci, test_groups in enumerate(combos):
        test_set = set(test_groups)
        train_groups = set(range(n_groups)) - test_set
        test_blocks = [group_bounds[g] for g in test_groups]

        train_recs = _purge_train(meta, train_groups, test_blocks, embargo)
        test_recs = [m["rec"] for m in meta if m["group"] in test_set]
        if not train_recs or not test_recs:
            logger.warning("Kombination %s/%s test=%s: tomt træ-/test-sæt efter purge — springes over.",
                           ci + 1, n_combos, test_groups)
            combo_preds.append({})
            continue

        tr_part, val_part = _time_sort_split(train_recs, config.VAL_RATIO)
        if stride > 1:
            tr_part = _subsample_train_stride(tr_part, stride)

        scaler = _fit_scaler(data_by_sym, tr_part, n_features)
        scaled_by_sym = _scale_all(data_by_sym, scaler, n_features)
        logger.info(
            "Kombination %s/%s test=%s: træn=%s (val=%s) → træner model…",
            ci + 1, n_combos, test_groups, len(tr_part), len(val_part),
        )
        model, best_val, _had_best, interrupted = _fit_lstm(
            scaled_by_sym, tr_part, val_part,
            device=device, seq_len=seq_len, n_features=n_features,
            log_prefix=f"[combo {ci + 1}/{n_combos}] ",
        )

        preds = _predict_recs(model, scaled_by_sym, test_recs, device, seq_len, n_features)
        combo_preds.append(preds)

        # OOS-metrik kun for denne kombinations test-grupper.
        m_best, m_avg, _eq = _simulate_metrics(preds, actual_all, min_syms)
        combo_rows.append({
            "kind": "combo",
            "combo": ci,
            "test_groups": "+".join(map(str, test_groups)),
            "n_train": len(tr_part),
            "n_test_preds": len(preds),
            "best_val_mse": round(float(best_val), 6) if np.isfinite(best_val) else None,
            "oos_sharpe_best": m_best.get("sharpe"),
            "oos_return_best_pct": m_best.get("total_return_pct"),
            "oos_maxdd_best_pct": m_best.get("max_drawdown_pct"),
            "oos_sharpe_avg": m_avg.get("sharpe"),
            "oos_return_avg_pct": m_avg.get("total_return_pct"),
        })
        del scaled_by_sym, model
        if interrupted:
            raise KeyboardInterrupt

    # --- Saml backtest-stier: hver gruppe testes i C(N-1,k-1) kombinationer ---
    combos_with_group: dict[int, list[int]] = {
        g: [ci for ci, tg in enumerate(combos) if g in tg] for g in range(n_groups)
    }
    phi = len(combos_with_group[0])
    path_rows: list[dict] = []
    path_equities: list[pd.Series] = []
    for p in range(phi):
        path_map: Dict[Tuple[pd.Timestamp, str], float] = {}
        for g in range(n_groups):
            ci = combos_with_group[g][p]
            for (ts, sym), v in combo_preds[ci].items():
                if date_to_group.get(ts) == g:
                    path_map[(ts, sym)] = v
        m_best, m_avg, eq_best = _simulate_metrics(path_map, actual_all, min_syms)
        if eq_best is not None:
            path_equities.append(eq_best)
        path_rows.append({
            "kind": "path",
            "path": p,
            "n_preds": len(path_map),
            "oos_sharpe_best": m_best.get("sharpe"),
            "oos_return_best_pct": m_best.get("total_return_pct"),
            "oos_maxdd_best_pct": m_best.get("max_drawdown_pct"),
            "oos_sharpe_avg": m_avg.get("sharpe"),
            "oos_return_avg_pct": m_avg.get("total_return_pct"),
        })

    # --- Fordelings-resumé over stierne (holdbarhedstallet) ---
    path_sharpes = [r["oos_sharpe_best"] for r in path_rows]
    path_returns = [r["oos_return_best_pct"] for r in path_rows]
    path_maxdd = [r["oos_maxdd_best_pct"] for r in path_rows]
    finite_sharpes = [s for s in path_sharpes if s is not None and np.isfinite(s)]
    p_sharpe_le_0 = (
        round(float(np.mean([s <= 0 for s in finite_sharpes])), 3) if finite_sharpes else None
    )
    distribution = {
        "path_sharpe_best": _dist(path_sharpes),
        "path_return_best_pct": _dist(path_returns),
        "path_maxdd_best_pct": _dist(path_maxdd),
        "prob_sharpe_le_0": p_sharpe_le_0,
    }
    logger.info("=== CPCV holdbarhed (best, OOS) ===")
    logger.info("  Sharpe pr. sti: %s", distribution["path_sharpe_best"])
    logger.info("  Afkast%% pr. sti: %s", distribution["path_return_best_pct"])
    logger.info("  P(Sharpe ≤ 0):  %s", p_sharpe_le_0)

    # --- Gem CSV + JSON ---
    now = datetime.now()
    run_id = f"cpcv_{start_year}_{end_year}_{now:%Y%m%d_%H%M%S}"
    if output_path is None:
        output_path = _BACKTEST_DIR / f"{run_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combo_rows + path_rows).to_csv(output_path, index=False)
    out = {
        "run_id": run_id,
        "timestamp_iso": now.isoformat(timespec="seconds"),
        "start_year": start_year,
        "end_year": end_year,
        "n_groups": n_groups,
        "k_test": k_test,
        "n_combinations": n_combos,
        "n_paths": phi,
        "embargo_days": embargo_days,
        "seq_len": seq_len,
        "min_symbols_per_day": min_syms,
        "group_bounds": [[str(s.date()), str(e.date())] for s, e in group_bounds],
        "distribution": distribution,
        "combos": combo_rows,
        "paths": path_rows,
        "limitations": _CPCV_LIMITATIONS,
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("CPCV gemt til %s og %s.", output_path, json_path)

    result = {
        "run_id": run_id,
        "distribution": distribution,
        "combos": combo_rows,
        "paths": path_rows,
        "path_equities": path_equities,
        "csv_path": output_path,
        "json_path": json_path,
    }
    if show_plot:
        try:
            _plot_cpcv(path_equities, distribution, start_year, end_year, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CPCV-plot sprunget over (%s).", exc)
    return result


def _plot_cpcv(
    path_equities: list[pd.Series],
    distribution: dict,
    start_year: int,
    end_year: int,
    run_id: str,
) -> None:
    """To paneler: OOS equity-stier (log) + boxplot af Sharpe-fordelingen pr. sti."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})

    for i, eq in enumerate(path_equities):
        if eq is None or eq.empty:
            continue
        ax1.plot(eq.index, eq.values, linewidth=1.1, alpha=0.8, label=f"Sti {i}")
    ax1.axhline(START_EQUITY, color="#cccccc", linewidth=1.0, zorder=0)
    ax1.set_yscale("log")
    sh = distribution.get("path_sharpe_best", {})
    ax1.set_title(
        f"CPCV {start_year}-{end_year} — OOS backtest-stier (genoptrænet pr. fold)\n"
        f"kørsel {run_id} — Sharpe median={sh.get('median')} [{sh.get('min')}..{sh.get('max')}], "
        f"P(Sharpe≤0)={distribution.get('prob_sharpe_le_0')}",
        fontsize=9,
    )
    ax1.set_xlabel("Handelsdato")
    ax1.set_ylabel("Porteføljeværdi (USD, log)")
    ax1.grid(True, which="both", alpha=0.3)
    if path_equities:
        ax1.legend(loc="upper left", fontsize=8)

    sharpes = [
        e.pipe(_metrics.daily_returns_from_equity).pipe(_metrics.sharpe)
        for e in path_equities if e is not None and len(e) >= 2
    ]
    if sharpes:
        ax2.boxplot(sharpes, vert=True, widths=0.6)
        ax2.scatter(np.ones(len(sharpes)), sharpes, color="#1f77b4", alpha=0.7, zorder=3)
    ax2.axhline(0.0, color="#d62728", linewidth=1.0, linestyle="--")
    ax2.set_title("Sharpe-fordeling\n(holdbarhed)", fontsize=9)
    ax2.set_ylabel("Annualiseret Sharpe (OOS)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks([])

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_cpcv_backtest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
