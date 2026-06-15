"""Træningsloop med train/val-split og gem af model + scaler.

Label: næste handelsdags open→close-afkast i procent (se targets_next_day_open_to_close_pct).
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stock_predictor import config
from stock_predictor.data_fetcher import fetch_daily_bars
from stock_predictor.feature_engineer import (
    FEATURE_COLUMNS,
    engineer_features,
    targets_next_day_open_to_close_pct,
)
from stock_predictor.model import DailyLSTM
from stock_predictor.torch_device import device_supports_amp, resolve_device

logger = logging.getLogger(__name__)


def _unwrap_model(m: nn.Module) -> nn.Module:
    """torch.compile wrapper har typisk _orig_mod — checkpoints skal matche DailyLSTM."""
    return m._orig_mod if hasattr(m, "_orig_mod") else m


def _model_n_outputs() -> int:
    """Antal output-neuroner: 1 (punkt-estimat) eller len(UNCERTAINTY_QUANTILES)."""
    if bool(getattr(config, "UNCERTAINTY_HEAD_ENABLED", False)):
        return len(getattr(config, "UNCERTAINTY_QUANTILES", (0.1, 0.5, 0.9)))
    return 1


class _PinballLoss(nn.Module):
    """Kvantil-/pinball-loss til usikkerheds-head. pred: (B, Q), target: (B, 1).

    Pr. kvantil τ: max(τ·(y−q), (τ−1)·(y−q)); middel over kvantiler og batch. Lærer
    hovedet at forudsige percentilerne, så q90−q10 bliver et datadrevet konfidensbånd.
    """

    def __init__(self, quantiles) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(list(quantiles), dtype=torch.float32).view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred  # (B, Q) via broadcast af (B,1)
        return torch.maximum(self.q * diff, (self.q - 1.0) * diff).mean()


def _make_criterion(device: torch.device) -> nn.Module:
    """Pinball-loss når usikkerheds-head er til; ellers den eksisterende HuberLoss."""
    if bool(getattr(config, "UNCERTAINTY_HEAD_ENABLED", False)):
        return _PinballLoss(getattr(config, "UNCERTAINTY_QUANTILES", (0.1, 0.5, 0.9))).to(device)
    return nn.HuberLoss(delta=float(getattr(config, "HUBER_DELTA", 1.0)))


def _crisis_sample_weights(train_recs: List["WindowRec"], data_by_sym: dict) -> torch.Tensor:
    """Sample-vægt pr. trænings-record ud fra VIX på vinduets slutdato (krise-oversampling).

    w = clip(vix / VIX_REF, 1, MAX_WEIGHT): rolige dage vægt ~1, krak-dage op til MAX_WEIGHT,
    så modellen ser stress-regimer oftere. VIX læses fra den rå feature-matrix (vix_close).
    """
    vix_ref = float(getattr(config, "CRISIS_OVERSAMPLE_VIX_REF", 20.0))
    max_w = float(getattr(config, "CRISIS_OVERSAMPLE_MAX_WEIGHT", 5.0))
    try:
        vix_idx = FEATURE_COLUMNS.index("vix_close")
    except ValueError:
        return torch.ones(len(train_recs), dtype=torch.double)
    weights = np.empty(len(train_recs), dtype=np.float64)
    for k, r in enumerate(train_recs):
        vix = float(data_by_sym[r.sym].matrix[r.end_i, vix_idx])
        if not np.isfinite(vix) or vix <= 0:
            vix = vix_ref
        weights[k] = min(max_w, max(1.0, vix / vix_ref))
    return torch.from_numpy(weights)


@dataclass
class SymbolData:
    """Engineerede features for ét symbol — gemt én gang, ikke pr. vindue."""

    matrix: np.ndarray   # (T, n_features) float32
    y: np.ndarray        # (T,) float32 — næste-dags open→close-afkast pr. række
    end_dates: np.ndarray  # (T,) datetime64 — slutdato pr. række


@dataclass
class WindowRec:
    """Letvægts-indeks for ét træningsvindue (ingen array-kopi)."""

    sym: str
    end_i: int            # sidste rækkeindeks (inkl.) i symbolets matrix
    y: float
    end_date: np.datetime64


def _engineer_all(symbol_bars: dict, seq_len: int) -> dict[str, SymbolData]:
    """Engineer features én gang pr. symbol; ingen vindues-kopier her."""
    out: dict[str, SymbolData] = {}
    for sym, ohlcv in symbol_bars.items():
        if ohlcv is None or ohlcv.empty:
            continue
        try:
            feats = engineer_features(ohlcv)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature engineering fejlede for %s: %s", sym, exc)
            continue
        if len(feats) < seq_len + 1:
            logger.warning("For få rækker efter features for %s.", sym)
            continue
        y_pct = targets_next_day_open_to_close_pct(ohlcv, feats.index)
        out[sym] = SymbolData(
            matrix=feats.to_numpy(dtype=np.float32),
            y=y_pct.to_numpy(dtype=np.float32),
            end_dates=feats.index.to_numpy(),
        )
    return out


def _build_index(data_by_sym: dict[str, SymbolData], seq_len: int) -> List[WindowRec]:
    """Byg ét letvægts-WindowRec pr. gyldigt vindue (features er allerede dropna'et)."""
    recs: List[WindowRec] = []
    for sym, d in data_by_sym.items():
        last_i = d.matrix.shape[0] - 2  # behøver næste-dags target → drop sidste række
        first_i = seq_len - 1
        if last_i < first_i:
            continue
        for i in range(first_i, last_i + 1):
            tgt = float(d.y[i])
            if not np.isfinite(tgt):
                continue
            recs.append(WindowRec(sym=sym, end_i=i, y=tgt, end_date=d.end_dates[i]))
    return recs


def _time_sort_split(recs: List[WindowRec], val_ratio: float) -> Tuple[List[WindowRec], List[WindowRec]]:
    if not recs:
        return [], []
    ordered = sorted(recs, key=lambda r: r.end_date)
    if len(ordered) <= 3:
        return ordered, []
    n_val = max(1, int(len(ordered) * val_ratio))
    if n_val >= len(ordered):
        n_val = max(1, len(ordered) // 5)
    train_set = ordered[:-n_val]
    val_set = ordered[-n_val:]
    if not train_set:
        return ordered, []
    return train_set, val_set


def _subsample_train_stride(recs: List[WindowRec], stride: int) -> List[WindowRec]:
    """Behold hvert ``stride``'te vindue PR. SYMBOL (sorteret på end_i).

    Med SEQ_LEN-overlap er nabovinduer næsten identiske; en stride>1 skærer de redundante
    gradient-skridt fra uden at flytte tids-grænsen. ``stride`` <= 1 => uændret.
    """
    if stride <= 1:
        return recs
    by_sym: dict[str, List[WindowRec]] = {}
    for r in recs:
        by_sym.setdefault(r.sym, []).append(r)
    out: List[WindowRec] = []
    for rs in by_sym.values():
        rs.sort(key=lambda r: r.end_i)
        out.extend(rs[::stride])
    return out


def _fit_scaler(
    data_by_sym: dict[str, SymbolData],
    train_recs: List[WindowRec],
    n_features: int,
) -> RobustScaler:
    """Fit RobustScaler på unikke træningsrækker (ingen val/fremtids-lækage).

    Hvert symbols rækker op til dets seneste trænings-vindue indgår én gang —
    i modsætning til den gamle vindues-stak, der vægtede rækker efter overlap.
    """
    max_train_end: dict[str, int] = {}
    for r in train_recs:
        if r.end_i > max_train_end.get(r.sym, -1):
            max_train_end[r.sym] = r.end_i
    chunks = [data_by_sym[sym].matrix[: end_i + 1] for sym, end_i in max_train_end.items()]
    fit_rows = np.concatenate(chunks, axis=0)
    scaler = RobustScaler()
    scaler.fit(fit_rows.reshape(-1, n_features))
    return scaler


def _scale_all(
    data_by_sym: dict[str, SymbolData],
    scaler: RobustScaler,
    n_features: int,
) -> dict[str, np.ndarray]:
    """Skalér hvert symbols fulde matrix én gang → float32 (rækker lagres kun én gang)."""
    return {
        sym: scaler.transform(d.matrix.reshape(-1, n_features)).astype(np.float32)
        for sym, d in data_by_sym.items()
    }


class WindowDataset(Dataset):
    """Lazy dataset: slicer vinduer ud af de pr.-symbol-skalerede matricer pr. batch.

    Rækker lagres én gang i ``scaled_by_sym``; kun det enkelte batch-vindue
    materialiseres ad gangen, så RAM-forbruget er ~antal rækker (ikke ~vinduer×seq_len).
    """

    def __init__(self, scaled_by_sym: dict[str, np.ndarray], recs: List[WindowRec], seq_len: int) -> None:
        self.scaled = scaled_by_sym
        self.recs = recs
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.recs[i]
        window = self.scaled[r.sym][r.end_i - self.seq_len + 1 : r.end_i + 1]
        xt = torch.from_numpy(np.ascontiguousarray(window))
        yt = torch.tensor([r.y], dtype=torch.float32)
        return xt, yt


def train_model() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    cfg = config

    # Sørg for at makro-/olie-kolonnerne er materialiseret i cachen, før features bygges, så
    # `--train` er én kommando (oliedata behøver ikke en separat backfill). Fejl her må aldrig
    # blokere træning — uden olie-kolonner falder feature-laget neutralt tilbage.
    try:
        from stock_predictor.macro_features import ensure_macro_oil_cache

        ensure_macro_oil_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Makro-/olie-cache-refresh sprunget over: %s", exc)

    calendar_days = int(cfg.TRAINING_YEARS * 366 + cfg.FETCH_EXTRA_DAYS)

    # prefer_cache_only=False: tjek Alpaca for nye barer og opdatér cachen før træning;
    # falder tilbage til (evt. forældet) cache hvis API mangler nøgler eller fejler.
    fetch_result = fetch_daily_bars(
        cfg.ALPACA_API_KEY,
        cfg.ALPACA_SECRET_KEY,
        cfg.WATCHLIST,
        end=None,
        lookback_calendar_days=calendar_days,
        extra_buffer_days=0,
        prefer_cache_only=False,
    )
    bars = fetch_result.bars
    if fetch_result.cache_only:
        logger.info(
            "Træningsdata kun fra disk-cache (%s symboler; sidste bar %s).",
            len(bars),
            fetch_result.required_end,
        )
    if not bars:
        logger.error("Intet historisk data til træning. Afslutter.")
        return

    # Auto-opdatér nyere nyheds-sentiment (finBERT) ind i bars før features bygges.
    try:
        from stock_predictor.news_sentiment import refresh_watchlist

        refresh_watchlist(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, list(bars.keys()), bars)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News-sentiment refresh sprunget over: %s", exc)

    data_by_sym = _engineer_all(bars, cfg.SEQ_LEN)
    if not data_by_sym:
        logger.error("Ingen symboler med nok historik til træning. Afslutter.")
        return
    recs = _build_index(data_by_sym, cfg.SEQ_LEN)
    if not recs:
        logger.error("Ingen træningssekvenser kunne bygges. Afslutter.")
        return
    if len(recs) < 100:
        logger.warning(
            "Kun %s observationer til træning — resultat kan være ustabilt.",
            len(recs),
        )

    train_recs, val_recs = _time_sort_split(recs, cfg.VAL_RATIO)
    if not train_recs:
        logger.error("Tom træningsmængde efter split. Afslutter.")
        return
    # Valgfri trænings-stride (kun træningssættet; val beholdes i fuld tæthed).
    stride = int(getattr(cfg, "TRAIN_WINDOW_STRIDE", 1))
    if stride > 1:
        before = len(train_recs)
        train_recs = _subsample_train_stride(train_recs, stride)
        logger.info("Trænings-stride=%s: %s → %s vinduer.", stride, before, len(train_recs))
    # RobustScaler (median/IQR): robust over for de fede haler i log-afkast/volumen-delta.
    # Fit kun på unikke træningsrækker; skalér derefter hvert symbols matrix én gang.
    scaler = _fit_scaler(data_by_sym, train_recs, cfg.N_FEATURES)
    scaled_by_sym = _scale_all(data_by_sym, scaler, cfg.N_FEATURES)

    # Krise-oversampling: udregn sample-vægte (fra rå VIX) FØR data_by_sym frigives.
    crisis_weights = None
    if bool(getattr(cfg, "CRISIS_OVERSAMPLE_ENABLED", False)):
        crisis_weights = _crisis_sample_weights(train_recs, data_by_sym)
        logger.info(
            "Krise-oversampling til (VIX_REF=%.1f, MAX=%.1f): vægt-spænd %.2f..%.2f.",
            float(getattr(cfg, "CRISIS_OVERSAMPLE_VIX_REF", 20.0)),
            float(getattr(cfg, "CRISIS_OVERSAMPLE_MAX_WEIGHT", 5.0)),
            float(crisis_weights.min()), float(crisis_weights.max()),
        )

    del data_by_sym  # rå float32-matricer behøves ikke længere

    train_pref = getattr(cfg, "TRAIN_DEVICE", "auto")
    device = resolve_device(str(train_pref))
    logger.info("Træningsenhed: %s", device)

    use_amp = bool(getattr(cfg, "TRAIN_AMP", True)) and device_supports_amp(device)
    if bool(getattr(cfg, "TRAIN_AMP", True)) and not use_amp:
        logger.info("AMP slået fra (kræver CUDA).")

    scaler_amp: torch.amp.GradScaler | None
    if use_amp:
        scaler_amp = torch.amp.GradScaler("cuda")
    else:
        scaler_amp = None

    n_outputs = _model_n_outputs()
    model = DailyLSTM(
        n_features=config.N_FEATURES,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        n_outputs=n_outputs,
    ).to(device)
    if n_outputs > 1:
        logger.info(
            "Usikkerheds-head til: %s kvantiler %s (score=median, bånd=q90−q10).",
            n_outputs, tuple(getattr(config, "UNCERTAINTY_QUANTILES", ())),
        )

    if bool(getattr(cfg, "TORCH_COMPILE_TRAIN", False)):
        try:
            mode = str(getattr(cfg, "TORCH_COMPILE_MODE", "default"))
            model = torch.compile(model, mode=mode)  # type: ignore[assignment]
            logger.info("torch.compile aktiv (mode=%s).", mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.compile fejlede, fortsætter uden: %s", exc)

    criterion = _make_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    lr_sched_enabled = bool(getattr(cfg, "LR_SCHEDULER_ENABLED", True)) and bool(val_recs)
    scheduler = None
    if lr_sched_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(cfg, "LR_SCHEDULER_FACTOR", 0.5)),
            patience=int(getattr(cfg, "LR_SCHEDULER_PATIENCE", 5)),
            min_lr=float(getattr(cfg, "LR_SCHEDULER_MIN_LR", 1e-6)),
        )

    # Lazy host-side WindowDataset: rækker ligger én gang i scaled_by_sym; batches slices ud
    # og flyttes til device i loopet. pin_memory + non_blocking kun relevant på CUDA.
    pin_memory = device.type == "cuda"
    num_workers = int(getattr(cfg, "TRAIN_NUM_WORKERS", 0))

    train_ds = WindowDataset(scaled_by_sym, train_recs, cfg.SEQ_LEN)
    val_ds = WindowDataset(scaled_by_sym, val_recs, cfg.SEQ_LEN) if val_recs else None

    # Krise-oversampling => WeightedRandomSampler (stress-dage oftere); ellers almindelig shuffle.
    train_sampler = None
    if crisis_weights is not None and len(crisis_weights) == len(train_ds):
        train_sampler = WeightedRandomSampler(
            crisis_weights, num_samples=len(crisis_weights), replacement=True
        )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = (
        DataLoader(
            val_ds,
            batch_size=max(1, int(getattr(cfg, "VAL_BATCH_SIZE", cfg.BATCH_SIZE))),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if val_ds is not None
        else None
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = int(getattr(cfg, "EARLY_STOP_PATIENCE", 20))

    # Median-kolonne til retnings-/IC-metrik: kvantil-head => q50-kolonnen, ellers kolonne 0.
    if n_outputs > 1:
        _qs = list(getattr(cfg, "UNCERTAINTY_QUANTILES", (0.1, 0.5, 0.9)))
        median_col = min(range(len(_qs)), key=lambda i: abs(_qs[i] - 0.5))
    else:
        median_col = 0

    disk_best_val = float("inf")
    if cfg.MODEL_PATH.is_file():
        try:
            prev = torch.load(cfg.MODEL_PATH, map_location=device)
            if isinstance(prev, dict) and "best_val_mse" in prev:
                disk_best_val = float(prev["best_val_mse"])
                prev_n = int(prev["n_features"]) if "n_features" in prev else None
                prev_out = int(prev["n_outputs"]) if "n_outputs" in prev else 1
                # Skift i feature-antal ELLER output-antal (Huber<->kvantil) gør disk-tabet
                # usammenligneligt — nulstil baseline så den nye arkitektur kan gemmes.
                if (prev_n is not None and prev_n != int(cfg.N_FEATURES)) or prev_out != n_outputs:
                    disk_best_val = float("inf")
                    logger.warning(
                        "Checkpoint på disk har n_features=%s/n_outputs=%s men config har %s/%s — "
                        "behandler som ingen baseline (ny model gemmes hvis val OK).",
                        prev_n, prev_out, cfg.N_FEATURES, n_outputs,
                    )
                else:
                    logger.info(
                        "Eksisterende checkpoint: best_val_mse=%.6f (ny model skal slå dette for at gemmes).",
                        disk_best_val,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Kunne ikke læse tidligere checkpoint: %s", exc)

    training_interrupted = False
    for epoch in range(cfg.EPOCHS):
        try:
            model.train()
            total = 0.0
            bad_batch = False

            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler_amp is not None:
                    with torch.amp.autocast("cuda"):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                else:
                    pred = model(xb)
                    loss = criterion(pred, yb)

                if not torch.isfinite(loss):
                    logger.error("Ikke-endeligt loss — stopper træning (tjek data/features).")
                    bad_batch = True
                    break

                if use_amp and scaler_amp is not None:
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        logger.error("Ikke-endelig gradientnorm — stopper træning.")
                        bad_batch = True
                        break
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(torch.as_tensor(grad_norm)):
                        logger.error("Ikke-endelig gradientnorm — stopper træning.")
                        bad_batch = True
                        break
                    optimizer.step()

                total += float(loss.detach()) * xb.size(0)

            if bad_batch:
                break

            train_loss = total / max(1, len(train_ds))

            if val_dl is not None:
                model.eval()
                # Batchet val-forward via DataLoader: hele val-sættet i ét kald sprænger RAM
                # (aktiveringer ~ N*seq_len*hidden). Akkumulér vægtet MSE pr. batch.
                n_val = len(val_ds)
                sse = 0.0
                pred_chunks: list[np.ndarray] = []
                tgt_chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for xv, yv in val_dl:
                        xv = xv.to(device, non_blocking=pin_memory)
                        yv = yv.to(device, non_blocking=pin_memory)
                        if use_amp and device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                vp = model(xv)
                                bloss = float(criterion(vp, yv).item())
                        else:
                            vp = model(xv)
                            bloss = float(criterion(vp, yv).item())
                        sse += bloss * xv.size(0)
                        pred_chunks.append(vp.detach().float().cpu().numpy()[:, median_col])
                        tgt_chunks.append(yv.detach().float().cpu().numpy().reshape(-1))
                vloss = sse / max(1, n_val)

                # Retnings-træf og rank-IC (Pearson) på val — gør "mere præcis" målbart, ikke
                # blot tabet (som er pinball/Huber, ikke MSE, når usikkerheds-head er til).
                dir_acc = float("nan")
                val_ic = float("nan")
                if pred_chunks:
                    vp_all = np.concatenate(pred_chunks)
                    vt_all = np.concatenate(tgt_chunks)
                    m = np.isfinite(vp_all) & np.isfinite(vt_all)
                    if int(m.sum()) > 1:
                        vp_all, vt_all = vp_all[m], vt_all[m]
                        dir_acc = float(np.mean((vp_all > 0) == (vt_all > 0)) * 100.0)
                        if vp_all.std() > 1e-12 and vt_all.std() > 1e-12:
                            val_ic = float(np.corrcoef(vp_all, vt_all)[0, 1])

                if lr_sched_enabled and scheduler is not None:
                    scheduler.step(vloss)

                improved = vloss < best_val - 1e-12
                if improved:
                    best_val = vloss
                    core = _unwrap_model(model)
                    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %s/%s train_loss=%.6f val_loss=%.6f val_dir_acc=%.1f%% val_ic=%+.3f "
                    "lr=%.2e (patience %s/%s)",
                    epoch + 1,
                    cfg.EPOCHS,
                    train_loss,
                    vloss,
                    dir_acc,
                    val_ic,
                    cur_lr,
                    epochs_no_improve,
                    patience,
                )

                if epochs_no_improve >= patience:
                    logger.info("Early stopping: ingen val-forbedring i %s epochs.", patience)
                    break
            else:
                logger.info(
                    "Epoch %s/%s train_mse=%.6f (ingen val — ingen early stopping)",
                    epoch + 1,
                    cfg.EPOCHS,
                    train_loss,
                )
        except KeyboardInterrupt:
            logger.info(
                "Træning afbrudt (Ctrl+C) — forsøger at gemme bedste model hvis den slår disk ved start.",
            )
            training_interrupted = True
            break

    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state)

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": _unwrap_model(model).state_dict(),
        "n_features": config.N_FEATURES,
        "seq_len": config.SEQ_LEN,
        "hidden": config.LSTM_HIDDEN,
        "layers": config.LSTM_LAYERS,
        "dropout": config.DROPOUT,
        "n_outputs": n_outputs,
        "quantiles": list(getattr(config, "UNCERTAINTY_QUANTILES", ())) if n_outputs > 1 else None,
        "best_val_mse": float(best_val) if best_state is not None else float("inf"),
    }

    only_if_better = bool(getattr(cfg, "SAVE_MODEL_ONLY_IF_BETTER_THAN_DISK", True))
    if val_ds is None:
        only_if_better = False

    if training_interrupted:
        should_save = False
        if val_ds is None:
            logger.warning("Ctrl+C: ingen val-split — gemmer ikke.")
        elif best_state is None:
            logger.warning("Ctrl+C: intet bedste val-state endnu — gemmer ikke.")
        elif save_dict["best_val_mse"] < disk_best_val - 1e-12:
            should_save = True
        else:
            logger.warning(
                "Ctrl+C: gemmer ikke model: bedste val_mse=%.6f er ikke bedre end disk=%.6f.",
                save_dict["best_val_mse"],
                disk_best_val,
            )
    else:
        should_save = True
        if val_ds is not None and best_state is None:
            should_save = False
            logger.error("Intet valideret bedste state — gemmer ikke (træning afbrudt eller ingen val?).")
        elif val_ds is not None and only_if_better and best_state is not None:
            should_save = save_dict["best_val_mse"] < disk_best_val - 1e-12
            if not should_save:
                logger.warning(
                    "Gemmer ikke model: bedste val_mse=%.6f er ikke bedre end disk=%.6f.",
                    save_dict["best_val_mse"],
                    disk_best_val,
                )

    if should_save:
        torch.save(save_dict, cfg.MODEL_PATH)
        joblib.dump(scaler, cfg.SCALER_PATH)
        logger.info(
            "Model+scaler gemt (best_val_mse=%.6f) til %s",
            save_dict["best_val_mse"],
            cfg.MODEL_PATH,
        )

    if training_interrupted:
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Walk-forward genoptræning (Option 4C) — TUNG; flag-styret, GPU-anbefalet.
# ---------------------------------------------------------------------------
def _train_fold(
    scaled_by_sym: dict[str, np.ndarray],
    train_recs: List[WindowRec],
    device: torch.device,
    n_outputs: int,
    *,
    crisis_weights: torch.Tensor | None = None,
    epochs: int | None = None,
) -> DailyLSTM:
    """Træn én walk-forward-fold (kompakt: ingen val/early-stop) og returnér modellen.

    Genbruger WindowDataset + _make_criterion + krise-sampler. Holdt simpel med vilje —
    walk-forward kører mange folds, så hver fold er et fast antal epochs uden val-split.
    """
    n_epochs = int(epochs if epochs is not None else config.EPOCHS)
    pin = device.type == "cuda"
    ds = WindowDataset(scaled_by_sym, train_recs, config.SEQ_LEN)
    sampler = None
    if crisis_weights is not None and len(crisis_weights) == len(ds):
        sampler = WeightedRandomSampler(crisis_weights, num_samples=len(crisis_weights), replacement=True)
    dl = DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=sampler is None, sampler=sampler,
        num_workers=int(getattr(config, "TRAIN_NUM_WORKERS", 0)), pin_memory=pin,
    )
    model = DailyLSTM(
        n_features=config.N_FEATURES, hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS, dropout=config.DROPOUT, n_outputs=n_outputs,
    ).to(device)
    criterion = _make_criterion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    model.train()
    for _epoch in range(n_epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                logger.error("Walk-forward: ikke-endeligt loss — afbryder fold.")
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def train_model_walk_forward() -> dict | None:
    """Walk-forward: genoptræn pr. år på ekspanderende vindue og scor året out-of-sample.

    For hvert testår Y (fra første år + WALK_FORWARD_TRAIN_MIN_YEARS) trænes en ny model på
    alle records med slutdato < Y, og år Y scores OOS. De stitches til én equity-kurve og
    rapporteres pr. regime/år som den almindelige regime-backtest. TUNG (~mange træninger) —
    kør helst på GPU. Gemmer output/backtests/walkforward_*.csv/.json og viser graf.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = config
    if not bool(getattr(cfg, "WALK_FORWARD_ENABLED", False)):
        logger.warning("WALK_FORWARD_ENABLED=False — sæt flaget for at køre walk-forward.")
        return None

    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Lange historik fra cache (offline). 25 år dækker hele regime-spektret.
    lookback = int(25 * 366 + cfg.FETCH_EXTRA_DAYS)
    fetch_result = fetch_daily_bars(
        cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, cfg.WATCHLIST,
        end=None, lookback_calendar_days=lookback, extra_buffer_days=0, prefer_cache_only=True,
    )
    bars = fetch_result.bars
    if not bars:
        logger.error("Ingen barer fra cache til walk-forward.")
        return None
    try:
        from stock_predictor.news_sentiment import refresh_watchlist
        refresh_watchlist(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, list(bars.keys()), bars)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News-sentiment refresh sprunget over: %s", exc)

    data_by_sym = _engineer_all(bars, cfg.SEQ_LEN)
    recs = _build_index(data_by_sym, cfg.SEQ_LEN)
    if not recs:
        logger.error("Ingen records til walk-forward.")
        return None
    years = sorted({int(pd.Timestamp(r.end_date).year) for r in recs})
    min_year, max_year = years[0], years[-1]
    first_test = min_year + int(getattr(cfg, "WALK_FORWARD_TRAIN_MIN_YEARS", 8))
    step = int(getattr(cfg, "WALK_FORWARD_STEP_YEARS", 1))
    test_years = list(range(first_test, max_year + 1, step))
    if not test_years:
        logger.error("For kort historik til walk-forward (min %s år træning).", first_test - min_year)
        return None

    n_outputs = _model_n_outputs()
    device = resolve_device(str(getattr(cfg, "TRAIN_DEVICE", "auto")))
    logger.info(
        "Walk-forward %s OOS-år (%s..%s) på enhed %s — TUNG; %s træninger.",
        len(test_years), test_years[0], test_years[-1], device, len(test_years),
    )

    # Lazy import (backtest importerer ikke train → ingen cyklus).
    from stock_predictor import backtest as _bt

    pred_cols: dict[str, list[pd.Series]] = {}
    actual_cols: dict[str, list[pd.Series]] = {}
    for ti, Y in enumerate(test_years, 1):
        train_recs = [r for r in recs if int(pd.Timestamp(r.end_date).year) < Y]
        if len(train_recs) < 100:
            logger.warning("Fold %s (år %s): for få train-records (%s) — springes over.", ti, Y, len(train_recs))
            continue
        scaler = _fit_scaler(data_by_sym, train_recs, cfg.N_FEATURES)
        scaled = _scale_all(data_by_sym, scaler, cfg.N_FEATURES)
        crisis_weights = (
            _crisis_sample_weights(train_recs, data_by_sym)
            if bool(getattr(cfg, "CRISIS_OVERSAMPLE_ENABLED", False)) else None
        )
        logger.info("Fold %s/%s: træner til < %s (%s records) → scorer %s OOS …",
                    ti, len(test_years), Y, len(train_recs), Y)
        model = _train_fold(scaled, train_recs, device, n_outputs, crisis_weights=crisis_weights)
        for sym, ohlcv in bars.items():
            res = _bt._predict_symbol(model, scaler, device, cfg.N_FEATURES, cfg.SEQ_LEN, ohlcv, Y)
            if res is None:
                continue
            p, a, _band = res
            pred_cols.setdefault(sym, []).append(p)
            actual_cols.setdefault(sym, []).append(a)

    if not pred_cols:
        logger.error("Walk-forward gav ingen OOS-forudsigelser.")
        return None

    pred_df = pd.DataFrame({s: pd.concat(v).sort_index() for s, v in pred_cols.items()}).sort_index()
    actual_df = pd.DataFrame({s: pd.concat(v).sort_index() for s, v in actual_cols.items()}).reindex(pred_df.index)

    min_syms = int(getattr(cfg, "MIN_SYMBOLS_PER_DAY", 10))
    keep = pred_df.notna().sum(axis=1) >= min_syms
    pred_df, actual_df = pred_df.loc[keep], actual_df.loc[keep]
    if pred_df.empty:
        logger.error("Walk-forward: ingen dage med ≥%s symboler.", min_syms)
        return None

    daily_log, equities = _bt._simulate(pred_df, actual_df)
    records = _bt._regime_report(equities, daily_log, test_years[0], max_year, variant="walk_forward")

    from datetime import datetime as _dt
    now = _dt.now()
    run_id = f"walkforward_{test_years[0]}_{max_year}_{now:%Y%m%d_%H%M%S}"
    out_dir = _bt._BACKTEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_id}.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    import json as _json
    meta = {
        "run_id": run_id, "timestamp_iso": now.isoformat(timespec="seconds"),
        "mode": "walk_forward", "test_years": test_years,
        "train_min_years": int(getattr(cfg, "WALK_FORWARD_TRAIN_MIN_YEARS", 8)),
        "uncertainty_head": bool(getattr(cfg, "UNCERTAINTY_HEAD_ENABLED", False)),
        "crisis_oversample": bool(getattr(cfg, "CRISIS_OVERSAMPLE_ENABLED", False)),
        "n_features": cfg.N_FEATURES, "records": records,
    }
    csv_path.with_suffix(".json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Walk-forward gemt: %s (+ .json).", csv_path)
    try:
        _bt._plot_regime(equities, records, test_years[0], max_year, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke tegne walk-forward-graf: %s", exc)
    return {"run_id": run_id, "equities": equities, "records": records, "csv_path": csv_path}


if __name__ == "__main__":
    train_model()
