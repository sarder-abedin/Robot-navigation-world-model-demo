"""
test_pose_estimator_rpi.py – dead-reckoning of the robot's world pose.

Covers pose_estimator (framework-agnostic, no torch/PyQt): forward/slow/backup
translation along the heading, in-place TURN/REROUTE rotation, dt clamping,
STOP/idle no-op, angle wrapping and the config constructor. No hardware needed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import pose_estimator as pe


def _est(**kw):
    defaults = dict(forward_speed_mps=1.0, slow_speed_mps=0.5,
                    backup_speed_mps=0.5, turn_rate_dps=90.0)
    defaults.update(kw)
    return pe.PoseEstimator(**defaults)


def test_starts_at_origin():
    p = _est().pose
    assert (p.x_m, p.y_m, p.heading_deg) == (0.0, 0.0, 0.0)


def test_forward_advances_along_plus_y():
    est = _est(forward_speed_mps=1.0)
    p = est.update("FORWARD", "", 1.0)
    # 1 m/s × 1 s along heading 0 → +1 m in Y (forward), X unchanged.
    assert math.isclose(p.y_m, 1.0, abs_tol=1e-9)
    assert math.isclose(p.x_m, 0.0, abs_tol=1e-9)


def test_slow_uses_slow_speed():
    est = _est(slow_speed_mps=0.25)
    p = est.update("SLOW", "", 1.0)
    assert math.isclose(p.y_m, 0.25, abs_tol=1e-9)


def test_backup_reverses():
    est = _est(backup_speed_mps=0.4)
    p = est.update("BACKUP", "", 1.0)
    assert math.isclose(p.y_m, -0.4, abs_tol=1e-9)
    assert math.isclose(p.x_m, 0.0, abs_tol=1e-9)


def test_stop_is_noop():
    est = _est()
    est.update("FORWARD", "", 1.0)
    before = est.pose
    p = est.update("STOP", "", 1.0)
    assert (p.x_m, p.y_m, p.heading_deg) == (before.x_m, before.y_m, before.heading_deg)


def test_turn_right_increases_heading():
    est = _est(turn_rate_dps=90.0)
    p = est.update("TURN", "right", 1.0)
    assert math.isclose(p.heading_deg, 90.0, abs_tol=1e-9)


def test_turn_left_decreases_heading():
    est = _est(turn_rate_dps=90.0)
    p = est.update("TURN", "left", 1.0)
    assert math.isclose(p.heading_deg, -90.0, abs_tol=1e-9)


def test_reroute_blank_direction_defaults_left():
    est = _est(turn_rate_dps=30.0)
    p = est.update("REROUTE", "", 1.0)
    assert math.isclose(p.heading_deg, -30.0, abs_tol=1e-9)


def test_forward_after_right_turn_goes_plus_x():
    """Turn 90° right (heading +90 → facing +X), then FORWARD moves along +X."""
    est = _est(forward_speed_mps=1.0, turn_rate_dps=90.0)
    est.update("TURN", "right", 1.0)          # heading → +90
    p = est.update("FORWARD", "", 1.0)
    assert math.isclose(p.x_m, 1.0, abs_tol=1e-9)
    assert math.isclose(p.y_m, 0.0, abs_tol=1e-9)


def test_dt_is_clamped():
    """A huge dt (stalled pipeline) must not teleport the robot; it caps at MAX_DT_S."""
    est = _est(forward_speed_mps=1.0)
    p = est.update("FORWARD", "", 100.0)
    assert math.isclose(p.y_m, pe.MAX_DT_S, abs_tol=1e-9)


def test_nonpositive_dt_no_motion():
    est = _est()
    assert est.update("FORWARD", "", 0.0).y_m == 0.0
    assert est.update("FORWARD", "", -1.0).y_m == 0.0


def test_bad_dt_no_crash():
    est = _est()
    p = est.update("FORWARD", "", None)     # garbage dt → treated as 0
    assert p.y_m == 0.0


def test_heading_wraps_to_pm180():
    est = _est(turn_rate_dps=90.0)
    for _ in range(5):                       # 5 × 90° right = 450° → wraps to 90°
        est.update("TURN", "right", 1.0)
    assert math.isclose(est.pose.heading_deg, 90.0, abs_tol=1e-9)


def test_reset_reorigins():
    est = _est()
    est.update("FORWARD", "", 1.0)
    est.update("TURN", "right", 1.0)
    est.reset()
    p = est.pose
    assert (p.x_m, p.y_m, p.heading_deg) == (0.0, 0.0, 0.0)


def test_pose_property_returns_copy():
    """Mutating a returned pose must not corrupt the estimator's internal state."""
    est = _est()
    p = est.pose
    p.x_m = 999.0
    assert est.pose.x_m == 0.0


def test_action_enum_like_value():
    """Accepts an Action-like object exposing .value (as the pipeline passes)."""
    class FakeAction:
        value = "FORWARD"
    est = _est(forward_speed_mps=1.0)
    p = est.update(FakeAction(), "", 1.0)
    assert math.isclose(p.y_m, 1.0, abs_tol=1e-9)


def test_from_config_reads_governor_and_pose():
    cfg = {
        "decision": {"governor": {"forward_speed_mps": 0.3, "slow_speed_mps": 0.12}},
        "pose": {"turn_rate_dps": 60.0, "backup_speed_mps": 0.2},
    }
    est = pe.PoseEstimator.from_config(cfg)
    assert math.isclose(est.update("FORWARD", "", 1.0).y_m, 0.3, abs_tol=1e-9)
    est.reset()
    assert math.isclose(est.update("SLOW", "", 1.0).y_m, 0.12, abs_tol=1e-9)
    est.reset()
    assert math.isclose(est.update("BACKUP", "", 1.0).y_m, -0.2, abs_tol=1e-9)
    est.reset()
    assert math.isclose(est.update("TURN", "right", 1.0).heading_deg, 60.0, abs_tol=1e-9)


def test_from_config_empty_uses_defaults():
    est = pe.PoseEstimator.from_config({})
    # No crash; produces some forward motion with the default speed.
    assert est.update("FORWARD", "", 1.0).y_m > 0.0
