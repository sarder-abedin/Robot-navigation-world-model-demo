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

logger = logging.getLogger(__name__)

_MAX_DUTY = 4095


def _duty_to_fraction(value: int) -> float:
    return min(1.0, abs(int(value)) / _MAX_DUTY)


class tankMotor:
    def __init__(self, gpiochip: int = 0) -> None:
        from gpiozero import Motor
        from gpiozero.pins.lgpio import LGPIOFactory
        factory = LGPIOFactory(chip=gpiochip)
        self._left = Motor(forward=24, backward=23, pin_factory=factory)
        self._right = Motor(forward=5, backward=6, pin_factory=factory)
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
