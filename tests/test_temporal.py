"""
Tests for the temporal action recogniser.
"""

import pytest
import yaml
from src.temporal_action import (
    FrameObstacleState,
    MotionPattern,
    TemporalActionRecognizer,
)


@pytest.fixture
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _push_n(recognizer, state: FrameObstacleState, n: int) -> None:
    for _ in range(n):
        recognizer.push(state)


def test_empty_history_returns_uncertain(cfg):
    ta = TemporalActionRecognizer(cfg)
    result = ta.classify()
    assert result.pattern == MotionPattern.UNCERTAIN


def test_clear_path_detected(cfg):
    ta = TemporalActionRecognizer(cfg)
    clear = FrameObstacleState(False, False, 0.0, 0.5)
    _push_n(ta, clear, 10)
    result = ta.classify()
    assert result.pattern == MotionPattern.STATIC_CLEAR
    assert result.temporal_risk == 0.0


def test_approaching_obstacle(cfg):
    ta = TemporalActionRecognizer(cfg)
    window = cfg["temporal_action"]["window_size"]
    # Simulate an obstacle growing in size (approaching)
    for i in range(window):
        area = 0.02 + i * 0.008   # grows from 2% to ~10%
        state = FrameObstacleState(True, True, area, 0.5)
        ta.push(state)
    result = ta.classify()
    assert result.pattern == MotionPattern.APPROACHING
    assert result.temporal_risk >= 0.5


def test_clearing_obstacle(cfg):
    ta = TemporalActionRecognizer(cfg)
    window = cfg["temporal_action"]["window_size"]
    # First half: obstacle present; second half: clear
    half = window // 2
    for _ in range(half):
        ta.push(FrameObstacleState(True, True, 0.08, 0.5))
    for _ in range(window - half):
        ta.push(FrameObstacleState(False, False, 0.0, 0.5))
    result = ta.classify()
    assert result.pattern == MotionPattern.CLEARING


def test_crossing_obstacle(cfg):
    ta = TemporalActionRecognizer(cfg)
    window = cfg["temporal_action"]["window_size"]
    # Obstacle moves laterally (centroid sweeps left→right), area stays constant
    for i in range(window):
        cx = 0.1 + (i / window) * 0.8   # 0.1 → 0.9
        ta.push(FrameObstacleState(True, i > 3, 0.04, cx))
    result = ta.classify()
    assert result.pattern == MotionPattern.CROSSING


def test_blocking_obstacle(cfg):
    ta = TemporalActionRecognizer(cfg)
    # Large, centered, stationary obstacle
    block = FrameObstacleState(True, True, 0.12, 0.5)
    _push_n(ta, block, cfg["temporal_action"]["window_size"])
    result = ta.classify()
    assert result.pattern == MotionPattern.BLOCKING
    assert result.temporal_risk >= 0.7
