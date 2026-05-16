"""
Tests for the WorldModel module using the _StubEncoder (no GPU required).
"""

import numpy as np
import pytest
import yaml
from src.world_model import WorldModel, _cosine_sim, _sigmoid_scale


@pytest.fixture
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_buffer_not_ready_initially(cfg):
    wm = WorldModel(cfg)
    wm.load()
    result = wm.predict()
    assert result.buffer_ready is False
    assert result.predicted_risk == 0.0


def test_predict_after_buffer_fills(cfg):
    """After clip_length frames are pushed, predict() should return a result."""
    wm = WorldModel(cfg)
    wm.load()
    clip_len = cfg["world_model"]["clip_length"]
    dummy_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    for _ in range(clip_len):
        wm.push_frame(dummy_frame)
    result = wm.predict()
    assert result.buffer_ready is True
    assert 0.0 <= result.predicted_risk <= 1.0
    assert result.label in ("BLOCKED", "MIXED", "CLEAR", "UNKNOWN")


def test_cosine_sim_identical_vectors():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_sim(v, v) - 1.0) < 1e-6


def test_cosine_sim_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine_sim(v1, v2)) < 1e-6


def test_sigmoid_scale_midpoint():
    # At x=0 sigmoid returns 0.5
    assert abs(_sigmoid_scale(0.0) - 0.5) < 1e-6


def test_anchor_update(cfg):
    wm = WorldModel(cfg)
    wm.load()
    obs = [np.full((64, 64, 3), 80, dtype=np.uint8)] * 3
    clr = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
    wm.build_anchors(obs, clr)
    # After updating anchors, predict should still work
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    for _ in range(cfg["world_model"]["clip_length"]):
        wm.push_frame(dummy)
    result = wm.predict()
    assert result.buffer_ready is True
