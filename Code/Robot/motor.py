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
import math
import time

logger = logging.getLogger(__name__)

_MAX_DUTY = 4095


def _duty_to_fraction(value: int) -> float:
    return min(1.0, abs(int(value)) / _MAX_DUTY)


class tankMotor:
    def __init__(self, gpiochip: int = 0, soft_start: bool = True,
                 ramp_step: float = 0.08, ramp_pause: float = 0.03) -> None:
        from gpiozero import Motor
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory(chip=gpiochip)
        self._left = Motor(forward=24, backward=23, pin_factory=factory)
        self._right = Motor(forward=5, backward=6, pin_factory=factory)
        # Soft-start: ramp an ACCELERATING PWM change into small increments so the
        # motor inrush current doesn't spike and brown out the Pi (symptom: the Pi
        # drops off the network the instant it starts to drive). ramp_step is the
        # max per-increment jump (fraction of full scale); it MUST be below the
        # drive fraction (e.g. speed_full/4095) or soft-start never engages — the
        # old default 0.35 was above a ~0.27 crawl, so a hard 0→drive step slipped
        # through. A slowdown/STOP always applies instantly (never delay a stop).
        self._soft_start = soft_start
        self._ramp_step = max(0.02, float(ramp_step))
        self._ramp_pause = max(0.0, float(ramp_pause))
        self._last_l = 0.0   # last signed fraction applied
        self._last_r = 0.0
        logger.info("tankMotor initialised (gpiochip%d, soft_start=%s, ramp_step=%.2f, "
                    "ramp_pause=%.3fs)", gpiochip, soft_start, self._ramp_step, self._ramp_pause)

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
        target_mag = max(abs(tl), abs(tr))
        last_mag = max(abs(self._last_l), abs(self._last_r))
        # Ramp only when spinning UP or reversing (target magnitude ≥ current) — a
        # slowdown or STOP applies instantly so an emergency stop is never delayed.
        # Steps scale with the jump so each increment stays ≈ ramp_step (peak inrush
        # is set by the first, smallest increment), capped so the ramp adds ≤ ~0.15s.
        if self._soft_start and target_mag >= last_mag and biggest_jump > self._ramp_step:
            steps = max(2, min(6, math.ceil(biggest_jump / self._ramp_step)))
            for i in range(1, steps + 1):
                self._apply_signed(self._left, self._last_l + (tl - self._last_l) * i / steps)
                self._apply_signed(self._right, self._last_r + (tr - self._last_r) * i / steps)
                if i < steps and self._ramp_pause > 0:
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
