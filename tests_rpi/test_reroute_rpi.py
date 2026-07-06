"""
test_reroute_rpi.py – closed-loop, context-aware reroute (wait / turn / backup).

Drives DecisionFuser.decide() at high risk and checks the behaviour selection,
the WAIT timeout, the turn-until-clear spin guard, the legacy fallback, and the
BACKUP action's protocol wiring.
"""

import os
import sys
import time

import pytest
import yaml

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from decision import Action, DecisionFuser


@pytest.fixture
def cfg():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        return yaml.safe_load(f)


def _hi(f, pattern="BLOCKING", label="", dl=None, dc=None, dr=None):
    """A high-risk decision with the given motion/label/depth."""
    return f.decide(0.9, 0.9, 0.9, "BLOCKED", pattern,
                    obstacle_label=label, depth_left_m=dl, depth_center_m=dc, depth_right_m=dr)


def test_static_wall_turns_toward_open_side(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="BLOCKING", dl=3.0, dc=0.4, dr=1.0)
    assert r.action == Action.REROUTE and r.reroute_direction == "left"
    f2 = DecisionFuser(cfg, "predictive")
    r2 = _hi(f2, pattern="BLOCKING", dl=1.0, dc=0.4, dr=3.0)
    assert r2.reroute_direction == "right"


def test_crossing_obstacle_waits(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="CROSSING", label="person", dc=0.6)
    assert r.action == Action.STOP and "wait" in r.explanation.lower()


def test_clearing_obstacle_waits(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="CLEARING", label="person", dc=0.6)
    assert r.action == Action.STOP and "wait" in r.explanation.lower()


def test_approaching_person_not_close_waits(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="APPROACHING", label="person", dc=1.5, dl=1.0, dr=1.0)
    assert r.action == Action.STOP


def test_approaching_close_backs_up(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="APPROACHING", label="chair", dc=0.2, dl=1.0, dr=1.0)
    assert r.action == Action.BACKUP and r.reroute_direction == ""


def test_static_object_turns_not_waits(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="APPROACHING", label="chair", dc=1.5, dl=2.0, dr=0.5)
    assert r.action == Action.REROUTE and r.reroute_direction == "left"


def test_wait_times_out_and_escalates_to_turn(cfg):
    f = DecisionFuser(cfg, "predictive")
    # pretend we've been waiting longer than the timeout
    f._wait_since = time.monotonic() - (f._rr_wait_timeout + 1.0)
    r = _hi(f, pattern="CROSSING", label="person", dl=2.0, dc=0.4, dr=0.5)
    assert r.action == Action.REROUTE and r.reroute_direction == "left"


def test_turn_guard_stops_after_too_long(cfg):
    f = DecisionFuser(cfg, "predictive")
    f._turn_since = time.monotonic() - (f._rr_max_turn + 1.0)
    r = _hi(f, pattern="BLOCKING", dl=3.0, dc=0.4, dr=1.0)
    assert r.action == Action.STOP and "reassess" in r.explanation.lower()


def test_blocked_with_center_clearest_stops_not_turns(cfg):
    """If straight ahead is the most open direction, don't turn into a side wall —
    STOP and reassess."""
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="BLOCKING", dl=0.5, dc=3.0, dr=0.5)
    assert r.action == Action.STOP and "no clearer side" in r.explanation.lower()


def test_turn_only_when_side_clearly_more_open(cfg):
    f = DecisionFuser(cfg, "predictive")
    # a side beats centre by more than the margin → TURN toward it
    r = _hi(f, pattern="BLOCKING", dl=2.0, dc=0.4, dr=0.6)
    assert r.action == Action.REROUTE and r.reroute_direction == "left"


def test_depth_motion_state_from_close_wall():
    """A close obstacle in depth (no YOLO box) becomes a 'present, centered' state
    so the motion recogniser isn't blind to walls."""
    from temporal_action import depth_to_obstacle_state
    near = depth_to_obstacle_state(0.4, presence_range_m=1.5)
    assert near is not None and near.obstacle_present and near.in_center
    far = depth_to_obstacle_state(3.0, presence_range_m=1.5)
    assert far is None                       # nothing within range → STATIC_CLEAR
    assert depth_to_obstacle_state(None) is None
    # closer → larger pseudo-area (so the recogniser can see it "grow"/approach)
    assert depth_to_obstacle_state(0.3).area_frac > depth_to_obstacle_state(1.0).area_frac


def test_backup_capped_when_no_rear_progress(cfg):
    """The robot has no rear sensor, so a sustained BACKUP is capped → STOP."""
    f = DecisionFuser(cfg, "predictive")
    r1 = _hi(f, pattern="APPROACHING", label="chair", dc=0.2, dl=1.0, dr=1.0)
    assert r1.action == Action.BACKUP
    f._backup_since = time.monotonic() - (f._rr_backup_max + 1.0)   # backed up too long
    r2 = _hi(f, pattern="APPROACHING", label="chair", dc=0.2, dl=1.0, dr=1.0)
    assert r2.action == Action.STOP and "rear sensor" in r2.explanation.lower()


def test_blind_depth_still_decides_without_crash(cfg):
    f = DecisionFuser(cfg, "predictive")
    r = _hi(f, pattern="BLOCKING", dl=None, dc=None, dr=None)
    assert r.action == Action.REROUTE and r.reroute_direction == ""   # default spin


def test_avoidance_resets_when_path_clears(cfg):
    f = DecisionFuser(cfg, "predictive")
    _hi(f, pattern="CROSSING", label="person", dc=0.6)   # start waiting
    assert f._wait_since != 0.0
    f.decide(0.0, 0.0, 0.0, "CLEAR", "STATIC_CLEAR")     # low risk → forward
    assert f._wait_since == 0.0 and f._turn_since == 0.0


def test_legacy_closed_loop_off_uses_one_shot(cfg):
    cfg = dict(cfg); cfg["decision"] = dict(cfg["decision"])
    cfg["decision"]["reroute"] = dict(cfg["decision"]["reroute"]); cfg["decision"]["reroute"]["closed_loop"] = False
    f = DecisionFuser(cfg, "predictive")
    # legacy: crossing person would NOT wait — it reroutes (BLOCKED) or stops
    r = f.decide(0.9, 0.9, 0.9, "BLOCKED", "APPROACHING", clear_direction="RIGHT")
    assert r.action == Action.REROUTE and r.reroute_direction == "right"


# ── BACKUP protocol wiring ────────────────────────────────────────────────────

def test_execute_action_backup_calls_backup():
    import robot_control as rc

    class Ctl:
        def __init__(s): s.did = None
        def forward(s): pass
        def slow_forward(s): pass
        def stop(s): s.did = "stop"
        def backup(s): s.did = "backup"
    c = Ctl()
    rc.execute_action(c, Action.BACKUP)
    assert c.did == "backup"


def test_execute_action_backup_falls_back_to_stop_if_unsupported():
    import robot_control as rc

    class OldCtl:
        def __init__(s): s.did = None
        def forward(s): pass
        def slow_forward(s): pass
        def stop(s): s.did = "stop"
    c = OldCtl()
    rc.execute_action(c, Action.BACKUP)      # no backup() → safe fallback to stop
    assert c.did == "stop"


def test_tcp_controller_sends_backup():
    from robot_control import TCPRobotController

    class FakeConn:
        def __init__(s): s.msgs = []
        def send_aimove(s, a): s.msgs.append(a); return True
    conn = FakeConn()
    ctl = TCPRobotController({"robot": {"ultrasonic_stop_cm": 30.0}}, conn)
    ctl.backup()
    assert conn.msgs == ["BACKUP"]


# ── Hard-stop resets an in-progress avoidance maneuver ────────────────────────

def test_hard_stop_resets_avoidance_timers(cfg):
    """An ultrasonic hard-stop interrupting a WAIT must clear the avoidance
    timers, so when risk resumes the WAIT starts fresh instead of being
    'already timed out' while the robot was actually stopped."""
    f = DecisionFuser(cfg, "predictive")
    _hi(f, pattern="CROSSING", label="person", dc=0.6)     # begin WAIT
    assert f._wait_since != 0.0
    # Ultrasonic reflex fires for a frame (obstacle within stop distance).
    r = f.decide(0.9, 0.9, 0.9, "BLOCKED", "CROSSING", ultrasonic_risk=1.0)
    assert r.action == Action.STOP
    assert f._wait_since == 0.0 and f._turn_since == 0.0 and f._backup_since == 0.0


# ── RobotController maneuvers must not block the pipeline thread ───────────────

def _direct_cfg(cfg):
    c = dict(cfg); c["robot"] = dict(cfg["robot"])
    c["robot"].setdefault("reroute_direction", "left")
    return c


def test_direct_controller_reroute_is_non_blocking(cfg):
    """RobotController.reroute() must return immediately (timed spin runs in a
    worker thread) rather than sleeping on the caller's (pipeline) thread."""
    from robot_control import RobotController

    class FakeMotor:
        def __init__(s): s.calls = []
        def setMotorModel(s, l, r): s.calls.append((l, r))

    class FakeCar:
        def __init__(s): s.motor = FakeMotor()

    car = FakeCar()
    ctl = RobotController(_direct_cfg(cfg), car)
    t0 = time.monotonic()
    ctl.reroute("right")
    assert (time.monotonic() - t0) < 0.1          # returned without blocking
    ctl.stop()                                     # preempt the worker
    # A subsequent stop leaves the motors halted.
    assert car.motor.calls[-1] == (0, 0)
