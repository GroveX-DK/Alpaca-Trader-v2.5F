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
from stock_predictor.data_fetcher import fetch_daily_bars  # noqa: E402
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
    net = DailyLSTM(
        n_features=int(ckpt["n_features"]),
        hidden_size=int(ckpt["hidden"]),
        num_layers=int(ckpt["layers"]),
        dropout=float(ckpt["dropout"]),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    n_features_ckpt = int(ckpt["n_features"])
    seq_len_ckpt = int(ckpt.get("seq_len", config.SEQ_LEN))
    return net, scaler, device, n_features_ckpt, seq_len_ckpt


def _score_watchlist() -> dict:
    """Scor hele watchlisten på nyeste vindue → score_map pr. symbol.

    score: forudsagt næste dags open→close i procent. OHLCV hentes via fetch_daily_bars
    (samme sti som træning); ved tail-API-fejl kan trimmed cache bruges som fallback.
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

    score_map: dict[str, float] = {}

    for sym in config.WATCHLIST:
        ohlcv = bars.get(sym)
        if ohlcv is None or ohlcv.empty:
            logger.warning("Spring %s over (manglende data).", sym)
            continue
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
        score_map[sym] = float(out.reshape(-1)[0])

    if not score_map:
        raise RuntimeError("Alle symboler blev filtrede fra ved inference.")
    return score_map


def predict_rankings() -> Tuple[str, float, List[Tuple[str, float]]]:
    """Returnér (bedste_symbol, score, fuld_ranking [(symbol, score)] faldende).

    score = forudsagt næste dags open→close i pct.
    """
    score_map = _score_watchlist()
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    best_sym, best_score = ranked[0]
    return best_sym, best_score, ranked


__all__ = [
    "predict_rankings",
]
