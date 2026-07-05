"""
test_speed_governor_rpi.py – kinematic safe-speed governor (Code/Server).

Verifies the stopping-distance math and the FORWARD/SLOW/STOP mapping, including
the two properties that make it useful: it slows earlier as reaction latency
grows, and a nonzero target speed shortens the required braking distance.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

from decision import Action
from speed_governor import SpeedGovernor


def _gov(**over):
    g = {
        "forward_speed_mps": 0.35, "slow_speed_mps": 0.18, "max_decel_mps2": 0.6,
        "target_speed_mps": 0.0, "safety_margin_m": 0.10,
        "min_reaction_s": 0.2, "max_reaction_s": 3.0, "enabled": True,
    }
    g.update(over)
    return SpeedGovernor({"decision": {"governor": g}})


def test_stopping_distance_matches_formula():
    g = _gov()
    # d_stop = v*t + v^2/(2a) + margin  (target 0)
    expected = 0.35 * 1.0 + 0.35 ** 2 / (2 * 0.6) + 0.10
    assert abs(g.stopping_distance_m(0.35, 1.0) - expected) < 1e-9


def test_action_far_mid_close():
    g = _gov()
    assert g.max_action(2.0, 1.0) == Action.FORWARD     # plenty of room
    assert g.max_action(0.5, 1.0) == Action.SLOW        # can slow but not full-speed
    assert g.max_action(0.2, 1.0) == Action.STOP        # can't stop in time even slow


def test_higher_reaction_slows_earlier():
    g = _gov()
    # more latency → longer stopping distance → more cautious at the same distance
    assert g.stopping_distance_m(0.35, 2.0) > g.stopping_distance_m(0.35, 1.0)
    # at 0.6 m: fine with a fast pipeline, but SLOW/STOP once the AI is laggy
    assert g.max_action(0.6, 0.3) == Action.FORWARD
    assert _CAUTION(g.max_action(0.6, 2.0)) >= _CAUTION(Action.SLOW)


def test_target_speed_reduces_braking_distance():
    # only needing to slow to 0.2 m/s (not stop) shortens the stopping distance
    g0 = _gov(target_speed_mps=0.0)
    gt = _gov(target_speed_mps=0.2)
    assert gt.stopping_distance_m(0.35, 1.0) < g0.stopping_distance_m(0.35, 1.0)


def test_reaction_time_is_clamped():
    g = _gov(min_reaction_s=0.5, max_reaction_s=2.0)
    # below the floor is treated as the floor; above the cap as the cap
    assert g.stopping_distance_m(0.35, 0.0) == g.stopping_distance_m(0.35, 0.5)
    assert g.stopping_distance_m(0.35, 9.9) == g.stopping_distance_m(0.35, 2.0)


def _CAUTION(a):
    return {Action.FORWARD: 0, Action.SLOW: 1, Action.STOP: 2, Action.REROUTE: 2}[a]
