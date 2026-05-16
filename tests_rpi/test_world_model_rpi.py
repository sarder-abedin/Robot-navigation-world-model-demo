"""Tests for WorldModel using the _StubEncoder – no GPU required."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import numpy as np
import pytest
import yaml
from world_model import WorldModel, _cosine_sim, _sigmoid_scale


@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def dummy_frames(cfg, n=None):
    n = n or cfg["camera"]["clip_length"]
    s = cfg["world_model"]["input_size"]
    return [np.random.randint(0, 255, (s, s, 3), dtype=np.uint8) for _ in range(n)]


def test_not_ready_initially(cfg):
    wm = WorldModel(cfg)
    wm.load()
    result = wm.predict([])
    assert result.buffer_ready is False


def test_predicts_after_full_clip(cfg):
    wm = WorldModel(cfg)
    wm.load()
    clip_len = cfg["camera"]["clip_length"]
    # Force the run_every cadence to fire on first call
    wm._call_count = cfg["world_model"]["run_every_n_frames"] - 1
    frames = dummy_frames(cfg, clip_len)
    result = wm.predict(frames)
    assert result.buffer_ready is True
    assert 0.0 <= result.predicted_risk <= 1.0
    assert result.label in ("BLOCKED", "MIXED", "CLEAR", "UNKNOWN")


def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_sim(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_sim(v1, v2)) < 1e-6


def test_sigmoid_at_zero():
    assert abs(_sigmoid_scale(0.0) - 0.5) < 1e-6


def test_anchor_update(cfg):
    wm = WorldModel(cfg)
    wm.load()
    s = cfg["world_model"]["input_size"]
    obs = [np.full((s, s, 3), 80, dtype=np.uint8)] * 3
    clr = [np.zeros((s, s, 3), dtype=np.uint8)] * 3
    wm.build_anchors(obs, clr)
    # After anchor update, full clip prediction should still work
    wm._call_count = cfg["world_model"]["run_every_n_frames"] - 1
    result = wm.predict(dummy_frames(cfg))
    assert result.buffer_ready is True
