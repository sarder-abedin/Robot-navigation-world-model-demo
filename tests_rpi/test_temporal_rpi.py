"""Tests for the Raspberry Pi temporal action recogniser."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import yaml
import pytest
from temporal_action import (
    FrameObstacleState, MotionPattern, TemporalActionRecognizer,
)


@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def push_n(r, state, n):
    for _ in range(n):
        r.push(state)


def test_uncertain_initially(cfg):
    ta = TemporalActionRecognizer(cfg)
    assert ta.classify().pattern == MotionPattern.UNCERTAIN


def test_static_clear(cfg):
    ta = TemporalActionRecognizer(cfg)
    push_n(ta, FrameObstacleState(False, False, 0.0, 0.5), 10)
    r = ta.classify()
    assert r.pattern == MotionPattern.STATIC_CLEAR
    assert r.temporal_risk == 0.0


def test_approaching(cfg):
    ta = TemporalActionRecognizer(cfg)
    w = cfg["temporal_action"]["window_size"]
    for i in range(w):
        ta.push(FrameObstacleState(True, True, 0.02 + i * 0.008, 0.5))
    assert ta.classify().pattern == MotionPattern.APPROACHING


def test_clearing(cfg):
    ta = TemporalActionRecognizer(cfg)
    w = cfg["temporal_action"]["window_size"]
    h = w // 2
    for _ in range(h):
        ta.push(FrameObstacleState(True, True, 0.08, 0.5))
    for _ in range(w - h):
        ta.push(FrameObstacleState(False, False, 0.0, 0.5))
    assert ta.classify().pattern == MotionPattern.CLEARING


def test_crossing(cfg):
    ta = TemporalActionRecognizer(cfg)
    w = cfg["temporal_action"]["window_size"]
    for i in range(w):
        cx = 0.1 + (i / w) * 0.8
        ta.push(FrameObstacleState(True, i > 3, 0.04, cx))
    assert ta.classify().pattern == MotionPattern.CROSSING


def test_blocking(cfg):
    ta = TemporalActionRecognizer(cfg)
    push_n(ta, FrameObstacleState(True, True, 0.12, 0.5),
           cfg["temporal_action"]["window_size"])
    r = ta.classify()
    assert r.pattern == MotionPattern.BLOCKING
    assert r.temporal_risk >= 0.7
