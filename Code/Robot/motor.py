"""
motor.py – tankMotor wrapper for the Freenove FNK0077 robot (Pi side).

BCM GPIO pin assignments (Freenove FNK0077):
  Left  motor: forward=7,  backward=8
  Right motor: forward=11, backward=10

setMotorModel(left, right) accepts signed PWM duty values in [-2048, 2048].
Positive → forward, negative → reverse.  Zero → coast stop.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_DUTY = 2048


def _duty_to_fraction(value: int) -> float:
    return min(1.0, abs(int(value)) / _MAX_DUTY)


class tankMotor:
    def __init__(self, gpiochip: int = 4) -> None:
        from gpiozero import Motor
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory(chip=gpiochip)
        self._left = Motor(forward=7, backward=8, pin_factory=factory)
        self._right = Motor(forward=11, backward=10, pin_factory=factory)
        logger.info("tankMotor initialised (gpiochip%d)", gpiochip)

    def setMotorModel(self, left: int, right: int) -> None:
        self._apply(self._left, left)
        self._apply(self._right, right)

    def _apply(self, motor, value: int) -> None:
        fraction = _duty_to_fraction(value)
        if value > 0:
            motor.forward(fraction)
        elif value < 0:
            motor.backward(fraction)
        else:
            motor.stop()

    def close(self) -> None:
        try:
            self._left.stop()
            self._right.stop()
            self._left.close()
            self._right.close()
        except Exception as exc:
            logger.warning("Motor close error: %s", exc)
        logger.info("tankMotor closed")
