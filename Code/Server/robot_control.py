"""
robot_control.py – Wrapper around the existing Freenove tankMotor and Car.

This module DOES NOT replace any Freenove hardware code.  It is a thin adapter
that translates high-level navigation actions (FORWARD, SLOW, STOP, REROUTE)
into the calls the existing motor.py / car.py already understand:

    car.motor.setMotorModel(left: int, right: int)
    car.sonic.get_distance()   → cm

Motor range: -4095 … +4095 (gpiozero, PCB v1/v2, Pi 4/5)
  Forward:      setMotorModel( L,  L)   where L > 0
  Backward:     setMotorModel(-L, -L)
  Turn left:    setMotorModel(-L,  L)
  Turn right:   setMotorModel( L, -L)
  Stop:         setMotorModel( 0,  0)

In demo mode a MockMotor replaces the real gpiozero Motor so the pipeline
can be tested on any machine without GPIO access.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RobotController:
    """
    High-level motor controller that wraps car.motor (Freenove tankMotor).

    Accepts a live `car` object (Car instance from car.py) and translates
    action strings into motor commands.
    """

    def __init__(self, cfg: dict, car):
        r = cfg["robot"]
        self._speed_full = r["speed_full"]
        self._speed_slow = r["speed_slow"]
        self._reroute_secs = r["reroute_turn_seconds"]
        self._reroute_dir = r["reroute_direction"]
        self._sonic_stop_cm = r["ultrasonic_stop_cm"]
        self._use_sonic_guard = r.get("use_ultrasonic_guard", True)
        self._car = car

    # ── High-level commands ───────────────────────────────────────────────────

    def forward(self) -> None:
        if self._sonic_blocked():
            logger.info("Sonic guard: forward blocked – stopping instead")
            self.stop()
            return
        self._set(self._speed_full, self._speed_full)

    def slow_forward(self) -> None:
        if self._sonic_blocked():
            self.stop()
            return
        self._set(self._speed_slow, self._speed_slow)

    def stop(self) -> None:
        self._set(0, 0)

    def reroute(self) -> None:
        """Stop, turn to avoid obstacle, then resume slow forward."""
        self.stop()
        time.sleep(0.2)
        if self._reroute_dir == "left":
            self._set(-self._speed_slow, self._speed_slow)
        else:
            self._set(self._speed_slow, -self._speed_slow)
        time.sleep(self._reroute_secs)
        self.stop()

    def safe_stop(self) -> None:
        self._set(0, 0)
        logger.warning("SAFE STOP – motors halted")

    # ── Ultrasonic risk helper ────────────────────────────────────────────────

    def get_ultrasonic_risk(self) -> float:
        """
        Returns a risk score in [0,1] derived from the ultrasonic distance.

        Below ultrasonic_stop_cm → 1.0 (full stop risk).
        Linear falloff from 3× stop distance to stop distance.
        """
        if not self._use_sonic_guard or self._car is None:
            return 0.0
        try:
            d = self._car.sonic.get_distance()
            if d <= 0:
                return 0.0  # sensor error – ignore
            stop = self._sonic_stop_cm
            if d <= stop:
                return 1.0
            warn = stop * 3
            if d >= warn:
                return 0.0
            return float((warn - d) / (warn - stop))
        except Exception as exc:
            logger.debug("Ultrasonic read error: %s", exc)
            return 0.0

    # ── Low-level ─────────────────────────────────────────────────────────────

    def _set(self, left: int, right: int) -> None:
        try:
            self._car.motor.setMotorModel(left, right)
            logger.debug("Motor: L=%d R=%d", left, right)
        except Exception as exc:
            logger.error("Motor command failed: %s", exc)

    def _sonic_blocked(self) -> bool:
        return self._use_sonic_guard and self.get_ultrasonic_risk() >= 1.0


class MockRobotController:
    """Simulated robot for demo / headless testing (no GPIO)."""

    def __init__(self, cfg: dict):
        r = cfg["robot"]
        self._speed_full = r["speed_full"]
        self._speed_slow = r["speed_slow"]
        self._reroute_secs = r["reroute_turn_seconds"]
        self._reroute_dir = r["reroute_direction"]
        self.last_command = "NONE"

    def forward(self) -> None:
        self.last_command = f"FORWARD(L={self._speed_full}, R={self._speed_full})"
        logger.info("[MockRobot] %s", self.last_command)

    def slow_forward(self) -> None:
        self.last_command = f"SLOW(L={self._speed_slow}, R={self._speed_slow})"
        logger.info("[MockRobot] %s", self.last_command)

    def stop(self) -> None:
        self.last_command = "STOP"
        logger.info("[MockRobot] %s", self.last_command)

    def reroute(self) -> None:
        self.last_command = f"REROUTE({self._reroute_dir})"
        logger.info("[MockRobot] %s", self.last_command)

    def safe_stop(self) -> None:
        self.last_command = "SAFE_STOP"
        logger.warning("[MockRobot] SAFE_STOP")

    def get_ultrasonic_risk(self) -> float:
        return 0.0


def build_robot_controller(cfg: dict, car=None):
    """Factory: real controller in live mode, mock in demo mode."""
    if cfg.get("mode") == "live":
        if car is None:
            raise ValueError("car object required in live mode")
        return RobotController(cfg, car)
    return MockRobotController(cfg)


def execute_action(controller, action: str) -> None:
    """Map a Decision Action enum value to a controller method."""
    from decision import Action
    dispatch = {
        Action.FORWARD:  controller.forward,
        Action.SLOW:     controller.slow_forward,
        Action.STOP:     controller.stop,
        Action.REROUTE:  controller.reroute,
    }
    fn = dispatch.get(action)
    if fn:
        fn()
    else:
        logger.warning("Unknown action: %s", action)
