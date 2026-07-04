"""
test_motor_rpi.py – Unit tests for tankMotor soft-start (Code/Robot/motor.py).

gpiozero is stubbed so the motor logic runs without a Pi. The soft-start ramps
big PWM jumps (motor inrush → Pi brownout mitigation) but applies small/steady
changes instantly.
"""

import importlib.util
import os
import sys
import types

import pytest

_ROBOT_MOTOR = os.path.join(os.path.dirname(__file__), "..", "Code", "Robot", "motor.py")


class _FakeMotor:
    def __init__(self, *a, **k):
        self.calls = []

    def forward(self, s):
        self.calls.append(("f", round(s, 3)))

    def backward(self, s):
        self.calls.append(("b", round(s, 3)))

    def stop(self):
        self.calls.append(("s", 0))

    def close(self):
        pass


@pytest.fixture
def motor_mod():
    gz = types.ModuleType("gpiozero")
    gz.Motor = _FakeMotor
    pins = types.ModuleType("gpiozero.pins")
    lg = types.ModuleType("gpiozero.pins.lgpio")
    lg.LGPIOFactory = lambda chip=0: object()
    sys.modules["gpiozero"] = gz
    sys.modules["gpiozero.pins"] = pins
    sys.modules["gpiozero.pins.lgpio"] = lg
    # Load the ROBOT motor.py by explicit path — there is also a Code/Server/motor.py
    # (the Freenove server motor with a different signature), so a bare
    # `import motor` can resolve to the wrong one depending on sys.path order.
    spec = importlib.util.spec_from_file_location("robot_motor_under_test", _ROBOT_MOTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for m in ("gpiozero", "gpiozero.pins", "gpiozero.pins.lgpio"):
        sys.modules.pop(m, None)


def test_soft_start_ramps_big_jump(motor_mod):
    m = motor_mod.tankMotor(gpiochip=0, ramp_pause=0.0)
    m._left.calls.clear()
    m.setMotorModel(3000, 3000)          # stop → full: big jump
    steps = [c[1] for c in m._left.calls]
    assert len(steps) == 4               # ramped in 4 steps
    assert steps == sorted(steps)        # monotonically increasing
    assert steps[-1] > 0.7               # reaches ~full


def test_steady_and_small_changes_apply_instantly(motor_mod):
    m = motor_mod.tankMotor(gpiochip=0, ramp_pause=0.0)
    m.setMotorModel(3000, 3000)          # ramp once
    m._left.calls.clear()
    m.setMotorModel(3000, 3000)          # same → no ramp
    assert len(m._left.calls) == 1
    m._left.calls.clear()
    m.setMotorModel(2900, 2900)          # tiny change → no ramp
    assert len(m._left.calls) == 1


def test_forward_to_reverse_ramps(motor_mod):
    m = motor_mod.tankMotor(gpiochip=0, ramp_pause=0.0)
    m.setMotorModel(3000, 3000)
    m._left.calls.clear()
    m.setMotorModel(-3000, -3000)        # full swing → ramp, ends in reverse
    assert len(m._left.calls) == 4
    assert m._left.calls[-1][0] == "b"


def test_soft_start_disabled_applies_instantly(motor_mod):
    m = motor_mod.tankMotor(gpiochip=0, soft_start=False)
    m._left.calls.clear()
    m.setMotorModel(3000, 3000)
    assert len(m._left.calls) == 1


def test_zero_stops(motor_mod):
    m = motor_mod.tankMotor(gpiochip=0, soft_start=False)
    m._left.calls.clear()
    m.setMotorModel(0, 0)
    assert m._left.calls[-1][0] == "s"
