"""Inference: ranger watchlist ud fra sidste LOOKBACK features.

Score er forudsagt næste handelsdags intradag-afkast i pct: (close/open - 1) * 100
for dag t+1 efter feature-vinduets sidste dag t.

Data kommer fra fetch_daily_bars for hele WATCHLIST (OHLCV + Watchlist-metriker +
vol_annual_pct); standard cache slår inkrementel merge til (nye handelsdage hentes
fra API og gemmes i Parquet).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import (  # noqa: E402
    append_todays_open_row,
    fetch_daily_bars,
    fetch_todays_open,
)
from stock_predictor.feature_engineer import engineer_features  # noqa: E402
from stock_predictor.model import DailyLSTM  # noqa: E402
from stock_predictor.torch_device import resolve_device  # noqa: E402

logger = logging.getLogger(__name__)


def _load_bundle():
    ckpt_path = config.MODEL_PATH
    scaler_path = config.SCALER_PATH
    if not ckpt_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            "Manglende model eller scaler — kør træning med --train først.",
        )
    scaler = joblib.load(scaler_path)
    pref = str(getattr(config, "INFERENCE_DEVICE", "cpu"))
    device = resolve_device(pref)
    logger.debug("Inference device: %s", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    n_outputs_ckpt = int(ckpt.get("n_outputs", 1))
    net = DailyLSTM(
        n_features=int(ckpt["n_features"]),
        hidden_size=int(ckpt["hidden"]),
        num_layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
        n_outputs=n_outputs_ckpt,
    )
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    # Kvantil-niveauer (eller None) bæres med på modellen til median/bånd-udtræk.
    net._quantiles = ckpt.get("quantiles")
    n_features_ckpt = int(ckpt["n_features"])
    seq_len_ckpt = int(ckpt.get("seq_len", config.SEQ_LEN))
    return net, scaler, device, n_features_ckpt, seq_len_ckpt


def quantile_indices(quantiles) -> tuple[int, int, int]:
    """(median_idx, lav_idx, høj_idx) for et sæt kvantil-niveauer (fx 0.1/0.5/0.9)."""
    qs = list(quantiles)
    median_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
    lo_idx = min(range(len(qs)), key=lambda i: qs[i])
    hi_idx = max(range(len(qs)), key=lambda i: qs[i])
    return median_idx, lo_idx, hi_idx


def reduce_outputs(out: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """Reducér model-output (N, n_outputs) til (score, bånd) pr. række.

    Punkt-estimat (n_outputs=1): score = output, bånd = 0. Kvantil-head: score = median
    (q50), bånd = q_høj − q_lav (fx q90 − q10) — et datadrevet usikkerheds-mål til sizing.
    """
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    n_out = out.shape[1]
    if n_out == 1:
        return out[:, 0], np.zeros(out.shape[0], dtype=np.float64)
    quants = getattr(model, "_quantiles", None) or [
        (i + 1) / (n_out + 1) for i in range(n_out)
    ]
    m_i, lo_i, hi_i = quantile_indices(quants)
    return out[:, m_i], out[:, hi_i] - out[:, lo_i]


def _score_watchlist() -> Tuple[dict, dict]:
    """Scor hele watchlisten på nyeste vindue → (score_map, band_map) pr. symbol.

    score: forudsagt næste dags open→close i procent (median når usikkerheds-head er aktivt).
    band: q90 − q10 (0 for punkt-estimat). OHLCV hentes via fetch_daily_bars (samme sti som
    træning); ved tail-API-fejl kan trimmed cache bruges som fallback.
    """
    model, scaler, device, n_features_ckpt, ckpt_seq = _load_bundle()

    if ckpt_seq != config.SEQ_LEN:
        raise RuntimeError(
            f"Checkpoint seq_len ({ckpt_seq}) matcher ikke config.SEQ_LEN ({config.SEQ_LEN}). "
            "Juster LOOKBACK_DAYS/SEQ_LEN eller kør --train igen.",
        )

    calendar_days = int(config.INFERENCE_FETCH_CALENDAR_DAYS)

    fetch_result = fetch_daily_bars(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.WATCHLIST,
        lookback_calendar_days=calendar_days,
        extra_buffer_days=0,
        prefer_cache_only=False,
    )
    bars = fetch_result.bars
    if not bars:
        raise RuntimeError("Kunne ikke hente barrer til inference.")

    # Auto-opdatér nyere nyheds-sentiment (finBERT) ind i bars før features bygges.
    try:
        from stock_predictor.news_sentiment import refresh_watchlist

        refresh_watchlist(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, list(bars.keys()), bars)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News-sentiment refresh sprunget over: %s", exc)

    # Dagens open (kendt lige efter åbning) injiceres som open_{t+1} for sidste fulde bar,
    # så modellen får dagens åbnings-gap (next_open_gap) i vinduets sidste række. Hentes
    # én gang for hele watchlisten; manglende open => neutralt gap 0 (append-helperen).
    todays_open: dict[str, float] = {}
    if getattr(config, "OPEN_FEATURE_ENABLED", False):
        try:
            todays_open = fetch_todays_open(
                config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, list(bars.keys())
            )
            logger.info("Dagens open hentet for %s/%s symboler.", len(todays_open), len(bars))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dagens-open-hentning sprunget over: %s", exc)

    score_map: dict[str, float] = {}
    band_map: dict[str, float] = {}

    for sym in config.WATCHLIST:
        ohlcv = bars.get(sym)
        if ohlcv is None or ohlcv.empty:
            logger.warning("Spring %s over (manglende data).", sym)
            continue
        if getattr(config, "OPEN_FEATURE_ENABLED", False):
            ohlcv = append_todays_open_row(ohlcv, todays_open.get(sym))
        try:
            feats = engineer_features(ohlcv)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Features fejlede for %s: %s", sym, exc)
            continue
        if len(feats) < config.SEQ_LEN:
            logger.warning(
                "Ikke nok datapunkter efter warmup for %s "
                "(feats=%s, ohlcv=%s, krævet SEQ_LEN=%s, hentet %s kalenderdage).",
                sym,
                len(feats),
                len(ohlcv),
                config.SEQ_LEN,
                calendar_days,
            )
            continue

        tail = feats.iloc[-config.SEQ_LEN :]
        seq = tail.to_numpy(dtype=np.float64)
        if np.any(np.isnan(seq)):
            continue
        n_feat = int(seq.shape[1])
        if n_feat != n_features_ckpt:
            logger.warning(
                "Feature-antal (%s) matcher ikke checkpoint (%s) for %s — spring over.",
                n_feat,
                n_features_ckpt,
                sym,
            )
            continue
        flat = scaler.transform(seq.reshape(-1, n_feat))
        xt = torch.from_numpy(flat.reshape(1, config.SEQ_LEN, n_feat).astype(np.float32))
        xt = xt.to(device)
        with torch.no_grad():
            out = model(xt).detach().cpu().numpy()
        score, band = reduce_outputs(out, model)
        score_map[sym] = float(score[0])
        band_map[sym] = float(band[0])

    if not score_map:
        raise RuntimeError("Alle symboler blev filtrede fra ved inference.")
    return score_map, band_map


def predict_rankings() -> Tuple[str, float, List[Tuple[str, float]]]:
    """Returnér (bedste_symbol, score, fuld_ranking [(symbol, score)] faldende).

    score = forudsagt næste dags open→close i pct (median når usikkerheds-head er aktivt).
    """
    score_map, _band = _score_watchlist()
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    best_sym, best_score = ranked[0]
    return best_sym, best_score, ranked


def predict_rankings_detailed() -> List[Tuple[str, float, float]]:
    """Fuld ranking som [(symbol, score, bånd)] faldende efter score.

    ``bånd`` = q90 − q10 fra usikkerheds-head (0 for punkt-estimat-model). Bruges af den
    long/short-traderen til konfidens-/vol-baseret sizing.
    """
    score_map, band_map = _score_watchlist()
    rows = [(s, sc, float(band_map.get(s, 0.0))) for s, sc in score_map.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def predict_best_directional() -> Tuple[str, float, float, str]:
    """Retningsbestemt valg: navnet med STØRST forudsagt |bevægelse|.

    Returnér (symbol, score, bånd, side) hvor ``score`` er median-forudsigelsen (open→close %),
    ``bånd`` = q90 − q10 (0 for punkt-estimat) og ``side`` = "long" hvis score ≥ 0 ellers
    "short". Bruges af den retningsbestemte live-sti (DIRECTIONAL_LIVE_ENABLED).
    """
    score_map, band_map = _score_watchlist()
    best_sym = max(score_map, key=lambda s: abs(score_map[s]))
    score = float(score_map[best_sym])
    band = float(band_map.get(best_sym, 0.0))
    side = "long" if score >= 0.0 else "short"
    return best_sym, score, band, side


__all__ = [
    "predict_rankings",
    "predict_rankings_detailed",
    "predict_best_directional",
    "reduce_outputs",
    "quantile_indices",
]
