"""
ultrasonic.py – HC-SR04 ultrasonic sensor wrapper for the Freenove FNK0077.

Default pins match config_robot.yaml:
  trigger_pin: 27
  echo_pin: 22

get_distance() returns centimetres (float). Returns max_range_cm on timeout/no-echo.

Primary backend: manual lgpio trigger/echo timing — this is what Freenove uses
on Pi 5 (BCM2712), where gpiozero's software-timed DistanceSensor is unreliable
and often reports "no echo received" even with a working sensor.
Fallback: gpiozero DistanceSensor (older Pi / if lgpio is unavailable).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_MAX_RANGE_CM = 400.0
_SPEED_OF_SOUND_CM_S = 34300.0
# _ECHO_TIMEOUT_S = 0.04   # ~6.8 m round-trip cap; beyond this = no echo
_ECHO_TIMEOUT_S = (2 * _MAX_RANGE_CM) / _SPEED_OF_SOUND_CM_S # timeout = (2 * max_range) / speed_of_sound


class Ultrasonic:
    def __init__(self, trigger_pin: int = 27, echo_pin: int = 22, gpiochip: int = 0) -> None:
        self._trigger = trigger_pin
        self._echo = echo_pin
        self._backend = None
        self._lgpio = None
        self._chip = None
        self._sensor = None

        # Try manual lgpio first (reliable on Pi 5).
        try:
            import lgpio  # type: ignore
            self._lgpio = lgpio
            self._chip = lgpio.gpiochip_open(gpiochip)
            lgpio.gpio_claim_output(self._chip, trigger_pin)
            lgpio.gpio_claim_input(self._chip, echo_pin)
            lgpio.gpio_write(self._chip, trigger_pin, 0)
            self._backend = "lgpio"
            logger.info(
                "Ultrasonic sensor initialised (lgpio, trigger=%d echo=%d gpiochip%d)",
                trigger_pin, echo_pin, gpiochip,
            )
            return
        except Exception as exc:
            logger.warning("lgpio ultrasonic init failed (%s) – trying gpiozero", exc)

        # Fallback: gpiozero DistanceSensor.
        from gpiozero import DistanceSensor  # type: ignore
        from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore
        factory = LGPIOFactory(chip=gpiochip)
        self._sensor = DistanceSensor(
            echo=echo_pin,
            trigger=trigger_pin,
            max_distance=_MAX_RANGE_CM / 100.0,
            pin_factory=factory,
        )
        self._backend = "gpiozero"
        logger.info(
            "Ultrasonic sensor initialised (gpiozero, trigger=%d echo=%d gpiochip%d)",
            trigger_pin, echo_pin, gpiochip,
        )

    def get_distance(self) -> float:
        if self._backend == "lgpio":
            return self._read_lgpio()
        if self._backend == "gpiozero":
            try:
                return round(self._sensor.distance * 100.0, 1)
            except Exception as exc:
                logger.debug("Ultrasonic read error: %s", exc)
                return _MAX_RANGE_CM
        return _MAX_RANGE_CM

    def _read_lgpio(self) -> float:
        lg = self._lgpio
        chip = self._chip
        try:
            # 10 µs trigger pulse.
            lg.gpio_write(chip, self._trigger, 0)
            time.sleep(0.000002)
            lg.gpio_write(chip, self._trigger, 1)
            time.sleep(0.00001)
            lg.gpio_write(chip, self._trigger, 0)

            #deadline = time.time() + _ECHO_TIMEOUT_S # old code
            deadline = time.perf_counter() + _ECHO_TIMEOUT_S # new code
            # Wait for the echo to go high.
            #start = time.time() # old code
            start = time.perf_counter() # new code
            while lg.gpio_read(chip, self._echo) == 0:
                start = time.perf_counter() # new code. old code: start = time.time()
                if start > deadline:
                    return _MAX_RANGE_CM
            # Wait for the echo to go low.
            stop = time.perf_counter() # new code. old code: stop = time.time()
            while lg.gpio_read(chip, self._echo) == 1:
                stop = time.perf_counter() # new code. old code: stop = time.time()
                if stop > deadline:
                    return _MAX_RANGE_CM

            distance = (stop - start) * _SPEED_OF_SOUND_CM_S / 2.0
            return round(min(distance, _MAX_RANGE_CM), 1)
        except Exception as exc:
            logger.debug("Ultrasonic lgpio read error: %s", exc)
            return _MAX_RANGE_CM

    def close(self) -> None:
        try:
            if self._backend == "lgpio" and self._chip is not None:
                self._lgpio.gpiochip_close(self._chip)
            elif self._backend == "gpiozero" and self._sensor is not None:
                self._sensor.close()
        except Exception as exc:
            logger.warning("Ultrasonic close error: %s", exc)
        logger.info("Ultrasonic sensor closed")
