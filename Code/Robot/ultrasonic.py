"""
ultrasonic.py – HC-SR04 ultrasonic sensor wrapper for the Freenove FNK0077.

Default pins match config_robot.yaml:
  trigger_pin: 27
  echo_pin: 22

get_distance() returns centimetres (float).  Returns max_range_cm on timeout.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_RANGE_CM = 400.0


class Ultrasonic:
    def __init__(self, trigger_pin: int = 27, echo_pin: int = 22) -> None:
        from gpiozero import DistanceSensor
        self._sensor = DistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=_MAX_RANGE_CM / 100.0,
        )
        logger.info("Ultrasonic sensor initialised (trigger=%d echo=%d)", trigger_pin, echo_pin)

    def get_distance(self) -> float:
        try:
            dist_cm = self._sensor.distance * 100.0
            return round(dist_cm, 1)
        except Exception as exc:
            logger.warning("Ultrasonic read error: %s", exc)
            return _MAX_RANGE_CM

    def close(self) -> None:
        try:
            self._sensor.close()
        except Exception as exc:
            logger.warning("Ultrasonic close error: %s", exc)
        logger.info("Ultrasonic sensor closed")
