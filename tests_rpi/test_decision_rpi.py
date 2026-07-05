"""
Tests for the Raspberry Pi decision fusion module.
Runs without GPU or hardware.  Imports from Code/Server.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import pytest
import yaml
from decision import Action, DecisionFuser


@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def test_low_risk_forward(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR")
    assert r.action == Action.FORWARD


def test_high_risk_stops(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.95, 0.95, 0.95, "BLOCKED", "APPROACHING")
    assert r.action in (Action.STOP, Action.REROUTE)


def test_blocking_pattern_reroutes(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.85, 0.9, 0.9, "BLOCKED", "BLOCKING")
    assert r.action == Action.REROUTE


def test_vjepa2_early_warning(cfg):
    """V-JEPA 2 BLOCKED label alone should decelerate in predictive mode."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.05, 0.05, 0.05, "BLOCKED", "STATIC_CLEAR")
    assert r.action == Action.SLOW, f"Expected SLOW, got {r.action}"


def test_baseline_ignores_world_model(cfg):
    """Baseline mode must not react to V-JEPA 2 BLOCKED when detector is clear."""
    f = DecisionFuser(cfg, "baseline")
    r = f.decide(0.05, 0.0, 0.05, "BLOCKED", "STATIC_CLEAR")
    assert r.action == Action.FORWARD, f"Expected FORWARD, got {r.action}"


def test_ultrasonic_override(cfg):
    """Ultrasonic risk=1.0 must hard-stop the robot regardless of other signals."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR", ultrasonic_risk=1.0)
    assert r.action == Action.STOP


def test_ultrasonic_emergency_stop_beats_blocking(cfg):
    """ultrasonic_risk=1.0 must STOP even when BLOCKING pattern would normally REROUTE."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.85, 0.9, 0.9, "BLOCKED", "BLOCKING", ultrasonic_risk=1.0)
    assert r.action == Action.STOP, (
        f"Expected STOP (hard obstacle in sensor range) but got {r.action}"
    )


def test_hysteresis(cfg):
    # After high risk, a small sub-hysteresis drop should keep the action elevated.
    # Drop from 0.9 to 0.88 is 0.02 < hysteresis (0.05), so risk stays at 0.9.
    f = DecisionFuser(cfg, "predictive")
    f.decide(0.9, 0.9, 0.9, "BLOCKED", "APPROACHING")
    r = f.decide(0.88, 0.88, 0.88, "MIXED", "UNCERTAIN")
    # Smoothed risk should remain ≥ 0.88 due to hysteresis holding it at 0.9
    assert r.risk_score >= 0.88


def test_medium_risk_slows(cfg):
    f = DecisionFuser(cfg, "predictive")
    mid = (cfg["decision"]["low_risk_max"] + cfg["decision"]["medium_risk_max"]) / 2
    r = f.decide(mid, mid, 0.0, "MIXED", "UNCERTAIN")
    assert r.action == Action.SLOW


def test_ultrasonic_is_separate_not_fused(cfg):
    """A partial ultrasonic risk (warn zone) must NOT drive the action or the
    fused risk score — the ultrasonic is a separate hard-stop, only at >=1.0."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR", ultrasonic_risk=0.7)
    assert r.action == Action.FORWARD          # 0.7 < 1.0 → no hard stop
    assert r.risk_score == 0.0                 # ultrasonic not blended into AI risk


def test_vision_reroutes_on_wm_blocked(cfg):
    """High AI risk with a V-JEPA 2 BLOCKED label reroutes (vision-driven turn),
    even without a BLOCKING temporal pattern and without ultrasonic."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.9, 0.9, 0.9, "BLOCKED", "APPROACHING")
    assert r.action == Action.REROUTE


def test_high_risk_without_vision_block_stops(cfg):
    """High AI risk but no vision block signal → STOP (can't know where to turn)."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.9, 0.9, 0.9, "MIXED", "APPROACHING")
    assert r.action == Action.STOP


# ── Speed governor caps forward motion by stopping distance ───────────────────

def test_governor_caps_forward_to_slow_when_close(cfg):
    """Vision is clear (would be FORWARD) but the confirmed-clear distance only
    allows a safe SLOW → the governor downgrades to SLOW."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                 clear_distance_m=0.5, reaction_s=1.0)
    assert r.action == Action.SLOW


def test_governor_caps_to_stop_when_too_close(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                 clear_distance_m=0.2, reaction_s=1.0)
    assert r.action == Action.STOP


def test_governor_allows_forward_with_room(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                 clear_distance_m=3.0, reaction_s=1.0)
    assert r.action == Action.FORWARD


def test_governor_skipped_when_blind(cfg):
    """No metric distance (blind sensor) → governor does not act; FORWARD stands
    (the ultrasonic hard-stop / blind-hold handles the safety instead)."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                 clear_distance_m=None, reaction_s=1.0)
    assert r.action == Action.FORWARD


def test_governor_never_speeds_up(cfg):
    """A STOP from high vision risk is not turned into FORWARD by lots of room."""
    f = DecisionFuser(cfg, "predictive")
    r = f.decide(0.9, 0.9, 0.9, "MIXED", "APPROACHING",
                 clear_distance_m=10.0, reaction_s=0.2)
    assert r.action == Action.STOP


def test_governor_latency_forces_earlier_slowing(cfg):
    """Same distance, but a laggier pipeline (bigger reaction_s) is more cautious."""
    f_fast = DecisionFuser(cfg, "predictive")
    fast = f_fast.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                         clear_distance_m=0.6, reaction_s=0.2)
    f_slow = DecisionFuser(cfg, "predictive")
    slow = f_slow.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR",
                         clear_distance_m=0.6, reaction_s=2.0)
    order = {Action.FORWARD: 0, Action.SLOW: 1, Action.STOP: 2}
    assert order[slow.action] >= order[fast.action]
