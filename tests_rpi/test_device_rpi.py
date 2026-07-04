"""
test_device_rpi.py – device auto-selection for the heavy models (device_utils).

torch is not required: a stub torch module is installed so the CUDA→MPS→CPU
resolution logic can be exercised deterministically.
"""

import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))


def _install_stub_torch(cuda: bool, mps: bool):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    backends = types.ModuleType("torch.backends")
    backends.mps = types.SimpleNamespace(is_available=lambda: mps)
    t.backends = backends

    class _Dev:
        def __init__(self, name):
            self.type = name

        def __repr__(self):
            return f"device({self.type})"

    t.device = _Dev
    sys.modules["torch"] = t
    sys.modules.pop("device_utils", None)
    return importlib.import_module("device_utils")


@pytest.mark.parametrize("pref,cuda,mps,expected", [
    ("auto",  True,  True,  "cuda"),   # CUDA wins
    ("auto",  False, True,  "mps"),    # MPS when no CUDA (native Mac)
    ("auto",  False, False, "cpu"),    # CPU fallback (e.g. Docker on Mac)
    ("cuda",  True,  False, "cuda"),   # explicit + available
    ("cuda",  False, True,  "mps"),    # explicit CUDA missing → auto → MPS
    ("mps",   False, False, "cpu"),    # explicit MPS missing → CPU
    ("cpu",   True,  True,  "cpu"),    # explicit CPU honoured despite GPUs
    ("bogus", True,  False, "cuda"),   # unknown preference → auto
    (None,    False, False, "cpu"),    # None → auto → cpu
])
def test_resolve_device(pref, cuda, mps, expected):
    du = _install_stub_torch(cuda, mps)
    dev, name = du.resolve_device(pref)
    assert name == expected
    assert dev.type == expected


def test_is_gpu():
    du = _install_stub_torch(False, False)
    assert du.is_gpu("cuda") is True
    assert du.is_gpu("mps") is True
    assert du.is_gpu("cpu") is False


def teardown_module(module):
    # Don't leave the stub torch installed for other test modules.
    sys.modules.pop("torch", None)
    sys.modules.pop("device_utils", None)
