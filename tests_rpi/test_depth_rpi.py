"""
test_depth_rpi.py – depth free-space channel + reroute-direction protocol.

freespace_from_depth is a pure function tested with synthetic depth maps (no
model needed); the estimator stub path and the CMD_AIMOVE#REROUTE#<dir> wiring
are checked too.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

from depth_perception import DepthEstimator, freespace_from_depth


def _depth(left, center, right, h=60, w=90):
    d = np.zeros((h, w), np.float32)
    d[:, : w // 3] = left
    d[:, w // 3: 2 * w // 3] = center
    d[:, 2 * w // 3:] = right
    return d


def test_nearest_ahead_is_center_band():
    r = freespace_from_depth(_depth(5.0, 0.4, 2.0), path_band_frac=1.0)
    assert abs(r.clear_distance_m - 0.4) < 1e-6
    assert r.region_distances_m["LEFT"] == 5.0


def test_open_direction_is_the_clearer_side():
    assert freespace_from_depth(_depth(5.0, 0.4, 2.0), path_band_frac=1.0).clear_direction == "LEFT"
    assert freespace_from_depth(_depth(2.0, 0.4, 5.0), path_band_frac=1.0).clear_direction == "RIGHT"


def test_center_when_no_side_is_clearly_better():
    # sides only marginally better than centre → don't turn
    r = freespace_from_depth(_depth(1.1, 1.0, 1.1), path_band_frac=1.0, direction_margin_m=0.3)
    assert r.clear_direction == "CENTER"


def test_depth_clamped_to_max_range():
    r = freespace_from_depth(_depth(99.0, 99.0, 99.0), path_band_frac=1.0, max_range_m=5.0)
    assert r.clear_distance_m == 5.0


def test_path_band_focuses_lower_frame():
    # obstacle only in the TOP half; lower band (the path) stays clear
    d = np.full((60, 90), 5.0, np.float32)
    d[:30, :] = 0.3
    r = freespace_from_depth(d, path_band_frac=0.5)
    assert r.clear_distance_m > 4.0


def test_estimator_stub_reports_not_ready():
    est = DepthEstimator({"depth": {"enabled": True}})
    est._model = None
    res = est.estimate(np.zeros((10, 10, 3), np.uint8))
    assert res.buffer_ready is False and res.is_stub is True


# ── CMD_AIMOVE#REROUTE#<dir> protocol ─────────────────────────────────────────

def test_execute_action_reroute_passes_direction():
    import robot_control as rc

    class FakeCtl:
        def __init__(s): s.rr = None
        def reroute(s, direction=""): s.rr = direction
        def forward(s): pass
        def slow_forward(s): pass
        def stop(s): pass

    from decision import Action
    c = FakeCtl()
    rc.execute_action(c, Action.REROUTE, "right")
    assert c.rr == "right"


def test_tcp_controller_sends_reroute_with_direction():
    from robot_control import TCPRobotController

    class FakeConn:
        def __init__(s): s.msgs = []
        def send_aimove(s, action): s.msgs.append(action); return True

    cfg = {"robot": {"ultrasonic_stop_cm": 30.0, "use_ultrasonic_guard": True}}
    conn = FakeConn()
    ctl = TCPRobotController(cfg, conn)
    ctl.reroute("left")
    ctl.reroute("")            # no direction → plain REROUTE
    assert conn.msgs == ["REROUTE#LEFT", "REROUTE"]
