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
import os

logger = logging.getLogger(__name__)

# Reduce allocator fragmentation for the big transient activations (V-JEPA 2's
# masked attention). `expandable_segments` lets the caching allocator grow an
# existing segment instead of failing when free memory is fragmented — exactly the
# hint printed in the HIP/CUDA OOM message ("try setting expandable_segments:True").
# Set for both HIP (ROCm) and CUDA; harmless on CPU. Must be set before the first
# CUDA/HIP allocation, and via setdefault so an explicit user value still wins.
# (On some ROCm builds expandable_segments is unsupported and simply ignored with a
# one-line warning — that's fine.)
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# THE ROCm fix for V-JEPA 2's OOM: on AMD GPUs PyTorch's scaled_dot_product_attention
# ships the memory-efficient / flash kernels (AOTriton) DISABLED by default, so even
# with attn_implementation="sdpa" it falls back to the *math* backend, which
# materialises the full N×N attention matrix — the multi-GiB spike that OOMs V-JEPA 2
# (ViT-L over a 64-frame clip). Enabling AOTriton makes SDPA use the real fused
# kernels (no N×N materialisation), so the masked forward fits on the GPU. Ignored on
# CUDA/CPU. setdefault so an explicit user value wins. Must precede the first HIP use.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def _is_available(name: str) -> bool:
    import torch  # type: ignore
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    return name == "cpu"


def _select(name: str):
    """Return (torch.device, name), enabling the MPS CPU-fallback when picking MPS.

    V-JEPA 2 / VideoMAE / Depth-Anything use a few ops Metal doesn't implement;
    PYTORCH_ENABLE_MPS_FALLBACK=1 makes those ops run on CPU instead of raising
    NotImplementedError, so the model runs on the Apple GPU end-to-end. Set via
    setdefault so an explicit user override is respected.
    """
    import torch  # type: ignore
    if name == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device(name), name


def resolve_device(preference: str = "auto"):
    """
    Return (torch.device, name) for the given preference.

    preference: "auto" | "cuda" | "mps" | "cpu" (case-insensitive).
    Raises ImportError only if torch itself is missing (callers handle that).
    """
    import torch  # type: ignore  # noqa: F401  (fail fast if torch is missing)

    pref = (preference or "auto").strip().lower()

    # Honour an explicit, available request.
    if pref in ("cuda", "mps", "cpu") and _is_available(pref):
        return _select(pref)

    if pref not in ("auto", "cuda", "mps", "cpu"):
        logger.warning("Unknown device preference %r – using auto", preference)
    elif pref in ("cuda", "mps"):
        logger.warning("Requested device '%s' unavailable – falling back (auto). "
                       "MPS needs the server run natively on macOS (no Metal in Docker).", pref)

    # Auto order: CUDA → MPS → CPU.
    for name in ("cuda", "mps", "cpu"):
        if _is_available(name):
            return _select(name)
    return _select("cpu")


def is_gpu(name: str) -> bool:
    return name in ("cuda", "mps")


def resolve_dtype(precision: str, device_name: str):
    """Pick the autocast compute dtype for the heavy models.

    precision: "bf16" | "fp16" | "fp32" (case-insensitive; default bf16).
    Half precision is only applied on CUDA/ROCm — that's where it both halves the
    activation memory (fixing the V-JEPA 2 attention OOM) and speeds inference up.
    CPU and MPS keep float32: bf16/fp16 are unsupported or slow there, and CPU is
    only ever the OOM-fallback path where correctness matters more than speed.
    Returns a torch.dtype.
    """
    import torch  # type: ignore
    p = (precision or "bf16").strip().lower()
    if device_name != "cuda":
        return torch.float32
    if p in ("fp16", "float16", "half", "16"):
        return torch.float16
    if p in ("fp32", "float32", "full", "32", "none"):
        return torch.float32
    return torch.bfloat16      # default / "bf16"


def autocast_ctx(device_name: str, dtype):
    """Return a torch.autocast context for CUDA half precision, else a no-op.

    Wrapping a forward in this runs its ops in bf16/fp16 on the GPU (half the
    activation memory, ~2× faster) while leaving the stored weights in fp32, so
    the CPU OOM-fallback path needs no dtype juggling."""
    import contextlib
    import torch  # type: ignore
    if device_name == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def is_oom_error(exc: BaseException) -> bool:
    """True if exc is a CUDA/ROCm out-of-memory error (version-robust).

    torch.cuda.OutOfMemoryError exists on newer torch and is raised for HIP OOM
    too; on older builds an OOM surfaces as a plain RuntimeError whose message
    contains 'out of memory', so fall back to a string check."""
    import torch  # type: ignore
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", ())
    if isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
