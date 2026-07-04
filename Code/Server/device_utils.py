"""
device_utils.py – pick the best available torch device for the heavy models.

Resolution order for "auto":  CUDA → MPS (Apple Metal) → CPU.

Environment notes
─────────────────
- CUDA works inside Docker on a Linux + NVIDIA host with `--gpus all` and the
  nvidia-container-toolkit (e.g. an NVIDIA DGX).
- MPS (Apple unified-memory GPU) is only reachable when the server runs
  *natively* on macOS. Inside a Docker container on a Mac there is no Metal
  passthrough, so the container falls back to CPU — that is expected.
- CPU always works everywhere.

An explicit request ("cuda"/"mps"/"cpu") is honoured when available; if the
requested accelerator is missing it degrades gracefully to the auto order.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _is_available(name: str) -> bool:
    import torch  # type: ignore
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    return name == "cpu"


def resolve_device(preference: str = "auto"):
    """
    Return (torch.device, name) for the given preference.

    preference: "auto" | "cuda" | "mps" | "cpu" (case-insensitive).
    Raises ImportError only if torch itself is missing (callers handle that).
    """
    import torch  # type: ignore

    pref = (preference or "auto").strip().lower()

    # Honour an explicit, available request.
    if pref in ("cuda", "mps", "cpu") and _is_available(pref):
        return torch.device(pref), pref

    if pref not in ("auto", "cuda", "mps", "cpu"):
        logger.warning("Unknown device preference %r – using auto", preference)
    elif pref in ("cuda", "mps"):
        logger.warning("Requested device '%s' unavailable – falling back (auto)", pref)

    # Auto order: CUDA → MPS → CPU.
    for name in ("cuda", "mps", "cpu"):
        if _is_available(name):
            return torch.device(name), name
    return torch.device("cpu"), "cpu"


def is_gpu(name: str) -> bool:
    return name in ("cuda", "mps")
