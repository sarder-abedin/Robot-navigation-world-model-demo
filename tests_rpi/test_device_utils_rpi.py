"""
test_device_utils_rpi.py – dtype/autocast/OOM helpers for GPU memory management.

Covers device_utils.resolve_dtype (bf16/fp16/fp32 selection), autocast_ctx (CUDA
half precision vs no-op), and is_oom_error (version-robust CUDA/ROCm OOM
detection). Needs torch for the dtype constants but NOT a GPU — skips if torch is
absent (same convention as the other torch-dependent tests).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

torch = pytest.importorskip("torch", reason="torch not installed")

import device_utils as du


# ── resolve_dtype ─────────────────────────────────────────────────────────────

def test_resolve_dtype_bf16_on_cuda():
    assert du.resolve_dtype("bf16", "cuda") is torch.bfloat16


def test_resolve_dtype_fp16_on_cuda():
    assert du.resolve_dtype("fp16", "cuda") is torch.float16
    assert du.resolve_dtype("float16", "cuda") is torch.float16
    assert du.resolve_dtype("half", "cuda") is torch.float16


def test_resolve_dtype_fp32_on_cuda():
    assert du.resolve_dtype("fp32", "cuda") is torch.float32
    assert du.resolve_dtype("full", "cuda") is torch.float32


def test_resolve_dtype_default_is_bf16():
    assert du.resolve_dtype("", "cuda") is torch.bfloat16
    assert du.resolve_dtype(None, "cuda") is torch.bfloat16
    assert du.resolve_dtype("garbage", "cuda") is torch.bfloat16   # unknown → bf16


def test_resolve_dtype_cpu_and_mps_stay_fp32():
    # Half precision is only applied on CUDA/ROCm; CPU/MPS keep full precision.
    for dev in ("cpu", "mps"):
        assert du.resolve_dtype("bf16", dev) is torch.float32
        assert du.resolve_dtype("fp16", dev) is torch.float32


# ── autocast_ctx ──────────────────────────────────────────────────────────────

def test_autocast_ctx_noop_on_cpu():
    import contextlib
    ctx = du.autocast_ctx("cpu", torch.float32)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_ctx_noop_when_fp32_on_cuda():
    import contextlib
    # fp32 on CUDA needs no autocast.
    ctx = du.autocast_ctx("cuda", torch.float32)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_ctx_returns_autocast_on_cuda_half():
    # bf16/fp16 on CUDA → a real torch.autocast (constructing it needs no GPU).
    ctx = du.autocast_ctx("cuda", torch.bfloat16)
    assert isinstance(ctx, torch.autocast)


# ── is_oom_error ──────────────────────────────────────────────────────────────

def test_is_oom_error_on_runtime_oom_message():
    assert du.is_oom_error(RuntimeError("HIP out of memory. Tried to allocate 12 GiB"))
    assert du.is_oom_error(RuntimeError("CUDA out of memory"))


def test_is_oom_error_false_for_other_runtime_errors():
    assert not du.is_oom_error(RuntimeError("shape mismatch"))
    assert not du.is_oom_error(ValueError("bad value"))
    assert not du.is_oom_error(TypeError("unexpected kwarg"))


def test_is_oom_error_true_for_torch_oom_class():
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is None:
        pytest.skip("this torch build has no torch.cuda.OutOfMemoryError")
    assert du.is_oom_error(oom_cls("out of memory"))
