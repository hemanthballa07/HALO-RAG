"""Runtime device selection helpers."""

from __future__ import annotations

import importlib.util
import logging


logger = logging.getLogger(__name__)


def resolve_device(preferred: str | None = "auto") -> str:
    """Return an available PyTorch device for the requested preference."""
    requested = (preferred or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if requested == "auto":
            return "cpu"
        raise RuntimeError(f"PyTorch is required to use the {requested!r} device") from None

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    available = cuda_available if requested == "cuda" else mps_available
    if available:
        return requested

    logger.warning("Requested device %s is unavailable; falling back to CPU", requested)
    return "cpu"


def qlora_supported(device: str) -> bool:
    """Return whether 4-bit QLoRA can be used in the current environment."""
    return device.startswith("cuda") and importlib.util.find_spec("bitsandbytes") is not None
