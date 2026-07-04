"""
motor.py – tankMotor wrapper for the Freenove FNK0077 robot (Pi side).

BCM GPIO pin assignments (Freenove FNK0077 PCB v1/v2):
  Left  motor: forward=24, backward=23  (Motor driver IN1/IN2)
  Right motor: forward=5,  backward=6   (Motor driver IN3/IN4)

  NOTE: BCM 7/8/25 are the SERVO pins on this board — do NOT use them here.

setMotorModel(left, right) accepts signed PWM duty values in [-4095, 4095].
Positive → forward, negative → reverse.  Zero → coast stop.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_MAX_DUTY = 4095


def _duty_to_fraction(value: int) -> float:
    return min(1.0, abs(int(value)) / _MAX_DUTY)


class tankMotor:
    def __init__(self, gpiochip: int = 0, soft_start: bool = True,
                 ramp_step: float = 0.35, ramp_pause: float = 0.02) -> None:
        from gpiozero import Motor
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory(chip=gpiochip)
        self._left = Motor(forward=24, backward=23, pin_factory=factory)
        self._right = Motor(forward=5, backward=6, pin_factory=factory)
        # Soft-start: ramp big PWM jumps over a few steps so the motor inrush
        # current doesn't spike and brown out the Pi (which shows up as the Pi
        # dropping off the network the moment it starts to drive). A jump larger
        # than ramp_step (fraction of full scale) is split into 4 steps ~ramp_pause
        # apart; steady/small changes apply instantly.
        self._soft_start = soft_start
        self._ramp_step = ramp_step
        self._ramp_pause = ramp_pause
        self._last_l = 0.0   # last signed fraction applied
        self._last_r = 0.0
        logger.info("tankMotor initialised (gpiochip%d, soft_start=%s)", gpiochip, soft_start)

    @staticmethod
    def _signed_fraction(value: int) -> float:
        f = _duty_to_fraction(value)
        return f if value > 0 else (-f if value < 0 else 0.0)

    @staticmethod
    def _apply_signed(motor, frac: float) -> None:
        if frac > 0:
            motor.forward(min(1.0, frac))
        elif frac < 0:
            motor.backward(min(1.0, -frac))
        else:
            motor.stop()

    def setMotorModel(self, left: int, right: int) -> None:
        tl = self._signed_fraction(left)
        tr = self._signed_fraction(right)
        biggest_jump = max(abs(tl - self._last_l), abs(tr - self._last_r))
        if self._soft_start and biggest_jump > self._ramp_step:
            steps = 4
            for i in range(1, steps + 1):
                self._apply_signed(self._left, self._last_l + (tl - self._last_l) * i / steps)
                self._apply_signed(self._right, self._last_r + (tr - self._last_r) * i / steps)
                if i < steps:
                    time.sleep(self._ramp_pause)
        else:
            self._apply_signed(self._left, tl)
            self._apply_signed(self._right, tr)
        self._last_l, self._last_r = tl, tr

    def close(self) -> None:
        try:
            self._left.stop()
            self._right.stop()
            self._left.close()
            self._right.close()
        except Exception as exc:
            logger.warning("Motor close error: %s", exc)
        logger.info("tankMotor closed")
