"""Fælles valg af torch.device til træning og inference."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str) -> torch.device:
    """
    preference: "auto" | "cuda" | "mps" | "cpu"
    Ved utilgængelig enhed falder cuda/mps roligt tilbage til cpu.
    """
    p = (preference or "cpu").strip().lower()
    if p == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if p == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA forespurgt men ikke tilgængelig — bruger cpu.")
        return torch.device("cpu")

    if p == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS forespurgt men ikke tilgængelig — bruger cpu.")
        return torch.device("cpu")

    if p == "cpu":
        return torch.device("cpu")

    logger.warning("Ukendt device-preference %r — bruger cpu.", preference)
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    """AMP (autocast+GradScaler) er primært understøttet og testet for CUDA."""
    return device.type == "cuda"
