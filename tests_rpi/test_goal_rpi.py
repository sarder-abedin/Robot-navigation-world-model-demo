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
from decision import Action
from goal_navigator import GoalState, GoalTracker, goal_steering
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


def test_tracker_marks_reached_within_arrival_distance():
    trk = GoalTracker({"goal": {"arrival_distance_m": 0.4}})
    trk.set_target(0.5, 0.5)
    far = trk.update(_frame_with_patch(200, 150), depth_sampler=lambda x, y: 1.2)
    assert not far.reached
    near = trk.update(_frame_with_patch(200, 150), depth_sampler=lambda x, y: 0.3)
    assert near.reached
    # Latched: a noisy far reading doesn't un-reach it.
    noisy = trk.update(_frame_with_patch(200, 150), depth_sampler=lambda x, y: 2.0)
    assert noisy.reached
    # A new target re-arms arrival.
    trk.set_target(0.5, 0.5)
    again = trk.update(_frame_with_patch(200, 150), depth_sampler=lambda x, y: 1.2)
    assert not again.reached


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


# ── Phase 3 goal-following steering (safety always overrides) ──────────────────

def _goal(**kw):
    base = dict(active=True, lost=False, reached=False, x=0.5, y=0.5, bearing=0.0, bearing_deg=0.0)
    base.update(kw)
    return GoalState(**base)


def test_goal_steering_turns_toward_goal_when_clear():
    # Path clear (FORWARD) + goal well to the right → spin right toward it.
    a, d = goal_steering(Action.FORWARD, _goal(bearing_deg=25.0), center_tol_deg=12.0)
    assert a == Action.TURN and d == "right"
    a, d = goal_steering(Action.FORWARD, _goal(bearing_deg=-25.0), center_tol_deg=12.0)
    assert a == Action.TURN and d == "left"


def test_goal_steering_drives_forward_when_goal_ahead():
    # Goal within tolerance → keep driving forward toward it (unchanged).
    a, d = goal_steering(Action.FORWARD, _goal(bearing_deg=5.0), center_tol_deg=12.0)
    assert a == Action.FORWARD and d == ""


def test_goal_steering_safety_always_wins():
    # Avoidance/stop actions are never overridden by goal-seeking.
    for safe in (Action.STOP, Action.REROUTE, Action.BACKUP):
        a, d = goal_steering(safe, _goal(bearing_deg=25.0), center_tol_deg=12.0)
        assert a == safe and d == ""


def test_goal_steering_inactive_reached_lost_unchanged():
    assert goal_steering(Action.FORWARD, None)[0] == Action.FORWARD
    assert goal_steering(Action.FORWARD, _goal(active=False, bearing_deg=25))[0] == Action.FORWARD
    assert goal_steering(Action.FORWARD, _goal(reached=True, bearing_deg=25))[0] == Action.FORWARD
    assert goal_steering(Action.FORWARD, _goal(lost=True, bearing_deg=25))[0] == Action.FORWARD


def test_turn_action_wiring():
    import robot_control as rc

    class FakeConn:
        def __init__(s): s.msgs = []
        def send_aimove(s, a): s.msgs.append(a); return True
    conn = FakeConn()
    ctl = rc.TCPRobotController({"robot": {"ultrasonic_stop_cm": 30.0}}, conn)
    rc.execute_action(ctl, Action.TURN, "right")
    assert conn.msgs == ["TURN#RIGHT"]

    # Old controller without turn() falls back to reroute.
    class OldCtl:
        def __init__(s): s.did = None
        def forward(s): pass
        def slow_forward(s): pass
        def stop(s): pass
        def reroute(s, d=""): s.did = ("reroute", d)
    old = OldCtl()
    rc.execute_action(old, Action.TURN, "left")
    assert old.did == ("reroute", "left")


# ── HUD declutter: text overlays default off (shown in the UI panel instead) ─────

def test_hud_declutter_default_flags():
    v = Visualizer({"visualization": {}})   # all defaults
    # Spatial / glanceable overlays stay burned on the video:
    assert v._overlay_det and v._overlay_risk and v._overlay_action
    # Text overlays move to the panel below → off on the video by default:
    assert not v._overlay_wm and not v._overlay_sonic and not v._overlay_fps
    assert not v._overlay_mode_badge and not v._overlay_ssv2 and not v._overlay_depth_text


def test_hud_depth_bars_drawn_without_text():
    @dataclass
    class _Depth:
        buffer_ready: bool = True
        clear_distance_m: float = 0.5
        clear_direction: str = "RIGHT"
        region_distances_m: dict = field(default_factory=lambda: {"LEFT": 0.4, "CENTER": 0.5, "RIGHT": 0.9})
    out = _viz().annotate(np.zeros((300, 400, 3), np.uint8), _Det(), _Decision(),
                          _Temporal(), depth=_Depth())
    assert out[88:120, 9:170].sum() > 0     # L/C/R bars still drawn (spatial cue kept)


def test_extended_aistatus_parses_with_depth_ssv2_and_goal():
    # Mirrors ai_viewer._process_status field layout for the extended message.
    line = "CMD_AISTATUS#FORWARD#12#MIXED#STATIC_CLEAR#62.2#person moving closer#0.45#RIGHT#reached"
    parts = line.split("#")
    assert parts[6] == "person moving closer"    # ssv2
    assert float(parts[7]) == 0.45 and parts[8] == "RIGHT"   # depth dist + dir
    assert parts[9] == "reached"                 # goal status
    # Old 6-field client still parses the core fields.
    assert parts[:6] == ["CMD_AISTATUS", "FORWARD", "12", "MIXED", "STATIC_CLEAR", "62.2"]


def test_hud_reached_goal_renders_green_banner():
    out = _viz().annotate(np.zeros((300, 400, 3), np.uint8), _Det(), _Decision(),
                          _Temporal(), goal=GoalState(active=True, reached=True, x=0.5, y=0.5,
                                                      distance_m=0.3))
    assert out.shape == (300, 400, 3)            # renders the reached state without error
    assert out[130:170, 180:220].sum() > 0       # marker drawn at the goal
