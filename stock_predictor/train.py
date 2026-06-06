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
from torch.utils.data import DataLoader, TensorDataset

from stock_predictor import config
from stock_predictor.data_fetcher import fetch_daily_bars
from stock_predictor.feature_engineer import engineer_features, targets_next_day_open_to_close_pct
from stock_predictor.model import DailyLSTM
from stock_predictor.torch_device import device_supports_amp, resolve_device

logger = logging.getLogger(__name__)


def _unwrap_model(m: nn.Module) -> nn.Module:
    """torch.compile wrapper har typisk _orig_mod — checkpoints skal matche DailyLSTM."""
    return m._orig_mod if hasattr(m, "_orig_mod") else m


@dataclass
class Sample:
    sequence: np.ndarray  # (seq_len, n_features)
    y: float
    end_date: np.datetime64


def _build_samples(
    symbol_bars: dict,
    seq_len: int,
) -> List[Sample]:
    samples: List[Sample] = []
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

        vals = feats.to_numpy(dtype=np.float64)
        idxs = feats.index.to_numpy()

        last_i = len(feats) - 2
        first_i = seq_len - 1
        if last_i < first_i:
            continue

        for i in range(first_i, last_i + 1):
            tgt = float(y_pct.iloc[i])
            if np.isnan(tgt) or np.any(np.isnan(vals[i - seq_len + 1 : i + 1])):
                continue
            samples.append(
                Sample(
                    sequence=vals[i - seq_len + 1 : i + 1].copy(),
                    y=tgt,
                    end_date=idxs[i],
                )
            )
    return samples


def _time_sort_split(samples: List[Sample], val_ratio: float) -> Tuple[List[Sample], List[Sample]]:
    if not samples:
        return [], []
    ordered = sorted(samples, key=lambda s: s.end_date)
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


def _prepare_tensors(
    scaler: RobustScaler,
    trains: List[Sample],
    vals: List[Sample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xt = np.stack([t.sequence.reshape(-1) for t in trains], axis=0)
    scaler.fit(xt.reshape(-1, config.N_FEATURES))

    def seq_scaled(samps: List[Sample]) -> np.ndarray:
        out = []
        for s in samps:
            flat = scaler.transform(s.sequence.reshape(-1, config.N_FEATURES))
            out.append(flat.reshape(config.SEQ_LEN, config.N_FEATURES))
        return np.stack(out, axis=0).astype(np.float32)

    def ys(samps: List[Sample]) -> np.ndarray:
        return np.array([[s.y] for s in samps], dtype=np.float32)

    if not vals:
        x_tr = seq_scaled(trains)
        y_tr = ys(trains)
        return torch.from_numpy(x_tr), torch.from_numpy(y_tr), None, None

    x_tr = seq_scaled(trains)
    x_va = seq_scaled(vals)
    y_tr = ys(trains)
    y_va = ys(vals)
    return torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(x_va), torch.from_numpy(y_va)


def train_model() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    cfg = config

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

    samples = _build_samples(bars, cfg.SEQ_LEN)
    if not samples:
        logger.error("Ingen træningssekvenser kunne bygges. Afslutter.")
        return
    if len(samples) < 100:
        logger.warning(
            "Kun %s observationer til træning — resultat kan være ustabilt.",
            len(samples),
        )

    train_s, val_s = _time_sort_split(samples, cfg.VAL_RATIO)
    if not train_s:
        logger.error("Tom træningsmængde efter split. Afslutter.")
        return
    # RobustScaler (median/IQR): robust over for de fede haler i log-afkast/volumen-delta.
    scaler = RobustScaler()
    X_t, Y_t, X_v, Y_v = _prepare_tensors(scaler, train_s, val_s)

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

    model = DailyLSTM(
        n_features=config.N_FEATURES,
        hidden_size=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    if bool(getattr(cfg, "TORCH_COMPILE_TRAIN", False)):
        try:
            mode = str(getattr(cfg, "TORCH_COMPILE_MODE", "default"))
            model = torch.compile(model, mode=mode)  # type: ignore[assignment]
            logger.info("torch.compile aktiv (mode=%s).", mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.compile fejlede, fortsætter uden: %s", exc)

    criterion = nn.HuberLoss(delta=float(getattr(cfg, "HUBER_DELTA", 1.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    lr_sched_enabled = bool(getattr(cfg, "LR_SCHEDULER_ENABLED", True)) and X_v is not None
    scheduler = None
    if lr_sched_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(cfg, "LR_SCHEDULER_FACTOR", 0.5)),
            patience=int(getattr(cfg, "LR_SCHEDULER_PATIENCE", 5)),
            min_lr=float(getattr(cfg, "LR_SCHEDULER_MIN_LR", 1e-6)),
        )

    # CUDA: host-side Dataset med pin_memory + non_blocking batchesoverførsel (Performance Guide).
    train_cuda_from_host = device.type == "cuda"
    if train_cuda_from_host:
        pin_memory = True
        num_workers = int(getattr(cfg, "TRAIN_NUM_WORKERS", 0))
    else:
        X_t = X_t.to(device)
        Y_t = Y_t.to(device)
        if X_v is not None and Y_v is not None:
            X_v = X_v.to(device)
            Y_v = Y_v.to(device)
        pin_memory = False
        num_workers = int(getattr(cfg, "TRAIN_NUM_WORKERS", 0))
        if device.type != "cpu":
            num_workers = 0

    if train_cuda_from_host and X_v is not None and Y_v is not None:
        X_v = X_v.to(device)
        Y_v = Y_v.to(device)

    train_ds = TensorDataset(X_t, Y_t)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    patience = int(getattr(cfg, "EARLY_STOP_PATIENCE", 20))

    disk_best_val = float("inf")
    if cfg.MODEL_PATH.is_file():
        try:
            prev = torch.load(cfg.MODEL_PATH, map_location=device)
            if isinstance(prev, dict) and "best_val_mse" in prev:
                disk_best_val = float(prev["best_val_mse"])
                prev_n = int(prev["n_features"]) if "n_features" in prev else None
                if prev_n is not None and prev_n != int(cfg.N_FEATURES):
                    disk_best_val = float("inf")
                    logger.warning(
                        "Checkpoint på disk har n_features=%s men config har %s — "
                        "behandler som ingen baseline (ny model gemmes hvis val OK).",
                        prev_n,
                        cfg.N_FEATURES,
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
                if train_cuda_from_host:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
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

            if X_v is not None and Y_v is not None:
                model.eval()
                # Batchet val-forward: hele val-sættet i ét kald sprænger RAM
                # (aktiveringer ~ N*seq_len*hidden). Akkumulér vægtet MSE i chunks.
                val_bs = max(1, int(getattr(cfg, "VAL_BATCH_SIZE", cfg.BATCH_SIZE)))
                n_val = X_v.size(0)
                sse = 0.0
                with torch.no_grad():
                    for vi in range(0, n_val, val_bs):
                        xv = X_v[vi : vi + val_bs]
                        yv = Y_v[vi : vi + val_bs]
                        if use_amp and device.type == "cuda":
                            with torch.amp.autocast("cuda"):
                                vp = model(xv)
                                bloss = float(criterion(vp, yv).item())
                        else:
                            vp = model(xv)
                            bloss = float(criterion(vp, yv).item())
                        sse += bloss * xv.size(0)
                vloss = sse / max(1, n_val)

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
                    "Epoch %s/%s train_mse=%.6f val_mse=%.6f lr=%.2e (patience %s/%s)",
                    epoch + 1,
                    cfg.EPOCHS,
                    train_loss,
                    vloss,
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
        "best_val_mse": float(best_val) if best_state is not None else float("inf"),
    }

    only_if_better = bool(getattr(cfg, "SAVE_MODEL_ONLY_IF_BETTER_THAN_DISK", True))
    if X_v is None:
        only_if_better = False

    if training_interrupted:
        should_save = False
        if X_v is None:
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
        if X_v is not None and best_state is None:
            should_save = False
            logger.error("Intet valideret bedste state — gemmer ikke (træning afbrudt eller ingen val?).")
        elif X_v is not None and only_if_better and best_state is not None:
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


if __name__ == "__main__":
    train_model()
