"""
test_wm_oom_rpi.py – V-JEPA 2 GPU-OOM → sticky-CPU re-probe decision.

After a GPU OOM the world model runs on CPU and STAYS there (no per-tick GPU↔CPU
thrash), re-probing the GPU every N CPU forwards. This covers the pure decision
helper `_should_retry_gpu`. world_model imports only numpy at module load (torch
is lazy), so this needs no torch/GPU.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import world_model as wm


def test_no_retry_before_interval():
    assert wm._should_retry_gpu(0, 30) is False
    assert wm._should_retry_gpu(29, 30) is False


def test_retry_at_and_after_interval():
    assert wm._should_retry_gpu(30, 30) is True
    assert wm._should_retry_gpu(31, 30) is True


def test_zero_interval_never_retries():
    # 0 = stay on CPU until the process restarts (no periodic GPU re-probe).
    assert wm._should_retry_gpu(0, 0) is False
    assert wm._should_retry_gpu(1000, 0) is False


def test_negative_interval_never_retries():
    assert wm._should_retry_gpu(1000, -5) is False


def test_interval_of_one_retries_every_call():
    assert wm._should_retry_gpu(1, 1) is True
