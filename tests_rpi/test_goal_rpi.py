"""
test_goal_rpi.py – goal-point navigation: tracking (Phase 2) + HUD rendering.

Phase 2 tracks a user-selected goal across frames and reports bearing + depth on
the HUD (no motion). These tests cover the tracker (init/update/lost, bearing
sign), the depth sampler, and the HUD drawing of the goal marker/arrow/readout.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

import cv2
from goal_navigator import GoalState, GoalTracker
from visualization import Visualizer


@dataclass
class _Decision:
    action: str = "FORWARD"
    risk_score: float = 0.1
    world_model_label: str = "MIXED"


@dataclass
class _Det:
    boxes: list = field(default_factory=list)
    obstacle_in_center: bool = False
    closest_area: float = 0.0


@dataclass
class _Temporal:
    pattern: str = "STATIC_CLEAR"


def _viz():
    return Visualizer({"visualization": {
        "show_window": False, "overlay_detections": False, "overlay_risk_bar": False,
        "overlay_action": False, "overlay_world_model_label": False,
    }})


def _frame_with_patch(cx, cy, w=400, h=300, size=40):
    """A black frame with a distinctive white square centred at (cx,cy) pixels."""
    f = np.zeros((h, w, 3), np.uint8)
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    f[y0:y0 + size, x0:x0 + size] = 255
    # a bit of texture so template matching / CSRT has features
    f[y0 + 5:y0 + 15, x0 + 5:x0 + 15] = 120
    return f


# ── Tracker ───────────────────────────────────────────────────────────────────

def test_tracker_reports_center_bearing_zero():
    trk = GoalTracker({"goal": {"patch_frac": 0.15}})
    trk.set_target(0.5, 0.5)
    st = trk.update(_frame_with_patch(200, 150))
    assert st.active and not st.lost
    assert abs(st.bearing) < 0.05 and abs(st.bearing_deg) < 2.0


def test_tracker_bearing_sign_left_right():
    trk = GoalTracker()
    trk.set_target(0.8, 0.5)                     # right of centre
    st = trk.update(_frame_with_patch(320, 150))
    assert st.bearing > 0 and st.bearing_deg > 0
    trk2 = GoalTracker()
    trk2.set_target(0.2, 0.5)                    # left of centre
    st2 = trk2.update(_frame_with_patch(80, 150))
    assert st2.bearing < 0 and st2.bearing_deg < 0


def test_tracker_follows_moving_patch():
    trk = GoalTracker({"goal": {"patch_frac": 0.15}})
    trk.set_target(0.5, 0.5)
    trk.update(_frame_with_patch(200, 150))       # init
    st = trk.update(_frame_with_patch(240, 150))  # patch moved right
    assert st.active and not st.lost
    assert st.x > 0.5                             # tracked point followed it right


def test_tracker_goes_lost_when_target_vanishes():
    trk = GoalTracker({"goal": {"patch_frac": 0.15, "max_lost_frames": 3}})
    trk.set_target(0.5, 0.5)
    trk.update(_frame_with_patch(200, 150))       # init on the patch
    blank = np.zeros((300, 400, 3), np.uint8)     # patch gone
    last = None
    for _ in range(5):
        last = trk.update(blank)
    assert last.lost is True


def test_tracker_depth_sampler_used():
    trk = GoalTracker()
    trk.set_target(0.5, 0.5)
    st = trk.update(_frame_with_patch(200, 150), depth_sampler=lambda x, y: 1.7)
    assert st.distance_m == pytest.approx(1.7)


def test_cleared_tracker_is_inactive():
    trk = GoalTracker()
    trk.set_target(0.5, 0.5)
    trk.update(_frame_with_patch(200, 150))
    trk.clear()
    assert not trk.active
    assert trk.update(_frame_with_patch(200, 150)).active is False


# ── Depth per-pixel sampler ─────────────────────────────────────────────────────

def test_depth_at_norm_samples_map():
    from depth_perception import DepthEstimator
    est = DepthEstimator({"depth": {"enabled": False, "max_range_m": 5.0}})
    m = np.full((30, 40), 2.5, np.float32)
    m[0, 0] = 1.1            # top-left corner
    m[29, 39] = 3.3          # bottom-right corner
    est._last_depth_map = m
    assert est.depth_at_norm(0.0, 0.0) == pytest.approx(1.1, abs=0.01)
    assert est.depth_at_norm(1.0, 1.0) == pytest.approx(3.3, abs=0.01)
    # clamped to max_range_m
    est._last_depth_map = np.full((4, 4), 9.0, np.float32)
    assert est.depth_at_norm(0.5, 0.5) == pytest.approx(5.0)
    est._last_depth_map = None
    assert est.depth_at_norm(0.5, 0.5) is None


# ── HUD rendering ───────────────────────────────────────────────────────────────

def test_hud_draws_goal_marker_and_arrow():
    out = _viz().annotate(np.zeros((300, 400, 3), np.uint8), _Det(), _Decision(),
                          _Temporal(), goal=GoalState(active=True, x=0.75, y=0.4,
                                                       bearing=0.5, bearing_deg=16.0,
                                                       distance_m=1.4))
    patch = out[int(0.4 * 300) - 14:int(0.4 * 300) + 14, int(0.75 * 400) - 14:int(0.75 * 400) + 14]
    assert patch.sum() > 0                        # marker drawn at the goal
    assert out[150, 200:280].sum() > 0            # arrow from centre toward the goal


def test_hud_inactive_goal_draws_no_marker():
    # An inactive goal must not draw a marker where an active (0.5,0.5) one would.
    out = _viz().annotate(np.zeros((300, 400, 3), np.uint8), _Det(), _Decision(),
                          _Temporal(), goal=GoalState(active=False))
    centre = out[150 - 20:150 + 20, 200 - 20:200 + 20]
    assert centre.sum() == 0


def test_hud_lost_goal_renders_without_error():
    out = _viz().annotate(np.zeros((300, 400, 3), np.uint8), _Det(), _Decision(),
                          _Temporal(), goal=GoalState(active=True, lost=True, x=0.5, y=0.5))
    assert out.shape == (300, 400, 3)


def test_permille_roundtrip_precision():
    for nx in (0.0, 0.123, 0.5, 0.999, 1.0):
        assert abs(int(nx * 1000) / 1000.0 - nx) <= 0.001
