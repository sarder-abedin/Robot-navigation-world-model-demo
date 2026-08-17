"""
test_camera_stall_rpi.py – camera stall detection + auto-restart decision logic.

Covers the framework-agnostic parts of Code/Robot/camera.py: the StreamingOutput
frame/timestamp bookkeeping, the pure stall/restart decision helpers, and
Camera._frame_or_none (return the frame unless the encoder has stalled). No
picamera2 / OpenCV / hardware needed — camera.py imports those lazily, so the
module imports cleanly with just the stdlib.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Robot"))

import camera as cam


class FakeClock:
    """A hand-cranked monotonic clock so time-based logic is deterministic."""
    def __init__(self, t=0.0):
        self.t = float(t)
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


# ── StreamingOutput bookkeeping ───────────────────────────────────────────────

def test_output_initial_state():
    out = cam.StreamingOutput(clock=FakeClock())
    assert out.frame is None
    assert out.seq == 0
    assert out.last_write_ts is None
    assert out.seconds_idle() is None        # nothing written yet


def test_output_write_updates_frame_seq_and_timestamp():
    clk = FakeClock(100.0)
    out = cam.StreamingOutput(clock=clk)
    out.write(b"jpeg1")
    assert out.frame == b"jpeg1"
    assert out.seq == 1
    assert out.last_write_ts == 100.0
    assert out.seconds_idle() == 0.0
    clk.advance(1.5)
    assert out.seconds_idle() == 1.5
    out.write(b"jpeg2")
    assert out.frame == b"jpeg2" and out.seq == 2
    assert out.seconds_idle() == 0.0         # timer reset by the new write


def test_output_reset_timer():
    clk = FakeClock(10.0)
    out = cam.StreamingOutput(clock=clk)
    out.write(b"x")
    clk.advance(5.0)
    assert out.seconds_idle() == 5.0
    out.reset_timer()                        # grace window, no new frame
    assert out.seconds_idle() == 0.0
    assert out.frame == b"x"                 # frame itself unchanged


# ── Pure decision helpers ─────────────────────────────────────────────────────

def test_is_stale():
    assert cam._is_stale(None, 100.0, 2.5) is False       # never written
    assert cam._is_stale(100.0, 101.0, 2.5) is False      # 1.0s < 2.5s
    assert cam._is_stale(100.0, 103.0, 2.5) is True       # 3.0s > 2.5s
    assert cam._is_stale(100.0, 102.5, 2.5) is False      # exactly at bound → not stale


def test_should_restart():
    assert cam._should_restart(None, 2.5) is False        # no frame yet → don't restart
    assert cam._should_restart(0.0, 2.5) is False
    assert cam._should_restart(2.4, 2.5) is False
    assert cam._should_restart(3.0, 2.5) is True


# ── Camera._frame_or_none (constructed without touching hardware) ─────────────

def _camera(clk, stall=2.5):
    # Camera.__init__ builds no hardware — only start_stream() would.
    return cam.Camera(stall_timeout_s=stall, clock=clk)


def test_frame_or_none_returns_fresh_frame():
    clk = FakeClock(50.0)
    c = _camera(clk)
    assert c._frame_or_none(b"frame", last_ts=50.0) == b"frame"


def test_frame_or_none_none_when_no_frame():
    c = _camera(FakeClock())
    assert c._frame_or_none(None, last_ts=None) is None


def test_frame_or_none_none_when_stale():
    clk = FakeClock(50.0)
    c = _camera(clk, stall=2.5)
    clk.t = 54.0                              # 4s since the frame at t=50 → stale
    assert c._frame_or_none(b"frozen", last_ts=50.0) is None


def test_frame_or_none_fresh_within_timeout():
    clk = FakeClock(50.0)
    c = _camera(clk, stall=2.5)
    clk.t = 52.0                              # 2s < 2.5s → still fresh
    assert c._frame_or_none(b"frame", last_ts=50.0) == b"frame"


# ── Camera construction / config clamping ─────────────────────────────────────

def test_camera_defaults_and_clamping():
    c = cam.Camera()
    assert c._stall_timeout == 2.5
    assert c._watchdog_interval == 1.0
    assert c._running is False               # not streaming until start_stream()
    # Absurdly small values are clamped to sane floors.
    c2 = cam.Camera(stall_timeout_s=0.0, watchdog_interval_s=0.0)
    assert c2._stall_timeout >= 0.5
    assert c2._watchdog_interval >= 0.2


def test_camera_shares_clock_with_output():
    clk = FakeClock(7.0)
    c = cam.Camera(clock=clk)
    c._output.write(b"z")
    assert c._output.last_write_ts == 7.0
