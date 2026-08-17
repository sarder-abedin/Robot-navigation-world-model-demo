"""
test_robot_command_dispatch_rpi.py – the robot command dispatch is crash-proof.

A malformed PC→Pi command or a transient motor/GPIO error must NEVER propagate
out of the command loop and crash the robot client — it's logged and swallowed so
the client keeps serving commands and the link stays up. Covers
main_robot._dispatch_command (the switch) and _safe_dispatch (the guard).
main_robot imports only stdlib + yaml at module load, so no hardware is needed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Robot"))

import main_robot as mr


class FakeMotor:
    """Records setMotorModel calls."""
    def __init__(self):
        self.calls = []
    def setMotorModel(self, left, right):
        self.calls.append((left, right))


class RaisingMotor:
    """Simulates a flaky GPIO/motor driver — every call blows up."""
    def setMotorModel(self, left, right):
        raise RuntimeError("GPIO busy")


FULL, SLOW, RR = 1500, 800, 1.2


# ── normal dispatch sets the expected motor state ─────────────────────────────

def test_forward():
    m = FakeMotor()
    assert mr._safe_dispatch("CMD_AIMOVE#FORWARD", m, FULL, SLOW, RR) is False
    assert m.calls[-1] == (FULL, FULL)


def test_slow():
    m = FakeMotor()
    mr._safe_dispatch("CMD_AIMOVE#SLOW", m, FULL, SLOW, RR)
    assert m.calls[-1] == (SLOW, SLOW)


def test_stop():
    m = FakeMotor()
    mr._safe_dispatch("CMD_AIMOVE#STOP", m, FULL, SLOW, RR)
    assert m.calls[-1] == (0, 0)


def test_turn_left_and_right():
    m = FakeMotor()
    mr._safe_dispatch("CMD_AIMOVE#TURN#left", m, FULL, SLOW, RR)
    assert m.calls[-1] == (-SLOW, SLOW)
    mr._safe_dispatch("CMD_AIMOVE#TURN#right", m, FULL, SLOW, RR)
    assert m.calls[-1] == (SLOW, -SLOW)


def test_manual_motor():
    m = FakeMotor()
    mr._safe_dispatch("CMD_MOTOR#120#-120", m, FULL, SLOW, RR)
    assert m.calls[-1] == (120, -120)


def test_cmd_stop_and_aimode_stop_motors():
    m = FakeMotor()
    mr._safe_dispatch("CMD_STOP", m, FULL, SLOW, RR)
    assert m.calls[-1] == (0, 0)
    mr._safe_dispatch("CMD_AIMODE#2", m, FULL, SLOW, RR)
    assert m.calls[-1] == (0, 0)


def test_kill_returns_true():
    assert mr._safe_dispatch("CMD_KILL", FakeMotor(), FULL, SLOW, RR) is True
    assert mr._safe_dispatch("CMD_AIMOVE#FORWARD", FakeMotor(), FULL, SLOW, RR) is False


# ── crash-proofing: garbage + throwing motor must never raise ─────────────────

@pytest.mark.parametrize("cmd", [
    "", "GARBAGE", "CMD_AIMOVE", "CMD_AIMOVE#", "CMD_AIMOVE#NOPE",
    "CMD_MOTOR", "CMD_MOTOR#x#y", "CMD_MOTOR#1", "###", "CMD_",
    "CMD_MOTOR#99999999999999999999#0",
])
def test_garbage_commands_never_raise(cmd):
    # A FakeMotor with a valid interface — malformed commands must be swallowed.
    assert mr._safe_dispatch(cmd, FakeMotor(), FULL, SLOW, RR) is False


@pytest.mark.parametrize("cmd", [
    "CMD_AIMOVE#FORWARD", "CMD_AIMOVE#SLOW", "CMD_AIMOVE#STOP",
    "CMD_AIMOVE#TURN#left", "CMD_STOP", "CMD_AIMODE#2", "CMD_MOTOR#10#20",
])
def test_motor_errors_never_propagate(cmd):
    # THE guarantee: a throwing motor/GPIO driver must not crash the client.
    assert mr._safe_dispatch(cmd, RaisingMotor(), FULL, SLOW, RR) is False


def test_dispatch_command_itself_does_raise_on_motor_error():
    # The guard is doing real work: the UNGUARDED dispatch propagates the error.
    with pytest.raises(RuntimeError):
        mr._dispatch_command("CMD_AIMOVE#FORWARD", RaisingMotor(), FULL, SLOW, RR)


def test_none_motor_is_fine():
    # Demo / no-hardware: motor is None → mock-logged, never raises.
    for cmd in ("CMD_AIMOVE#FORWARD", "CMD_STOP", "CMD_MOTOR#1#2", "CMD_AIMODE#0"):
        assert mr._safe_dispatch(cmd, None, FULL, SLOW, RR) is False
