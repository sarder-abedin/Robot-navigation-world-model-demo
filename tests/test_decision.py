"""
Tests for the decision fusion module.

These tests run without any ML models or hardware – they only exercise the
risk weighting, hysteresis, and action-selection logic.
"""

import pytest
import yaml
from src.decision import Action, DecisionFuser


@pytest.fixture
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_low_risk_gives_forward(cfg):
    fuser = DecisionFuser(cfg, "predictive")
    result = fuser.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR")
    assert result.action == Action.FORWARD
    assert result.risk_score < cfg["decision"]["low_risk_max"]


def test_high_risk_stops(cfg):
    fuser = DecisionFuser(cfg, "predictive")
    result = fuser.decide(0.9, 0.9, 0.9, "BLOCKED", "APPROACHING")
    assert result.action in (Action.STOP, Action.REROUTE)
    assert result.risk_score > cfg["decision"]["medium_risk_max"]


def test_blocking_pattern_reroutes(cfg):
    fuser = DecisionFuser(cfg, "predictive")
    result = fuser.decide(0.8, 0.85, 0.85, "BLOCKED", "BLOCKING")
    assert result.action == Action.REROUTE


def test_world_model_early_warning_slows_forward(cfg):
    """
    V-JEPA 2 predicts BLOCKED but detector risk is still low.
    In predictive mode the robot should slow down proactively.
    """
    fuser = DecisionFuser(cfg, "predictive")
    result = fuser.decide(0.05, 0.05, 0.05, "BLOCKED", "STATIC_CLEAR")
    assert result.action == Action.SLOW, (
        f"Expected SLOW due to WM early warning, got {result.action}"
    )


def test_baseline_mode_ignores_world_model(cfg):
    """
    In baseline mode V-JEPA 2 weight is 0, so a BLOCKED WM label alone
    should NOT cause deceleration when detector risk is low.
    """
    fuser = DecisionFuser(cfg, "baseline")
    result = fuser.decide(0.05, 0.0, 0.05, "BLOCKED", "STATIC_CLEAR")
    assert result.action == Action.FORWARD, (
        f"Baseline should stay FORWARD when detector risk is low, got {result.action}"
    )


def test_hysteresis_prevents_oscillation(cfg):
    """
    After a high-risk reading, risk must drop by more than the hysteresis
    margin before the action changes back.
    """
    fuser = DecisionFuser(cfg, "predictive")
    fuser.decide(0.9, 0.9, 0.9, "BLOCKED", "APPROACHING")  # drive risk high
    # Small drop – should still be high due to hysteresis
    result = fuser.decide(0.55, 0.55, 0.55, "MIXED", "UNCERTAIN")
    # The smoothed risk should still be above medium threshold
    assert result.risk_score > cfg["decision"]["medium_risk_max"] * 0.8


def test_medium_risk_slows(cfg):
    fuser = DecisionFuser(cfg, "predictive")
    # Risk just above low_risk_max but below medium_risk_max
    mid = (cfg["decision"]["low_risk_max"] + cfg["decision"]["medium_risk_max"]) / 2
    result = fuser.decide(mid, mid, 0.0, "MIXED", "UNCERTAIN")
    assert result.action == Action.SLOW
