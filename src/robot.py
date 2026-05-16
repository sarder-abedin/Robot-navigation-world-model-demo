"""
robot.py – Freenove tank robot motor control.

The Freenove ESP32 tank robot communicates over a serial connection.
Commands are plain-text strings in the format expected by the Freenove
firmware (https://github.com/Freenove/Freenove_Tank_Robot_Kit_for_ESP32).

Command format used here:
  CMD_MOTOR#<left_fwd>#<left_back>#<right_fwd>#<right_back>#\n

Where each value is 0-4095 (12-bit PWM).

The RobotController exposes high-level actions (forward, slow, stop, turn)
and translates them into the correct motor command bytes.

In demo mode a MockRobot is used instead, which logs commands to stdout
so the full pipeline can be tested without physical hardware.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RobotController:
    """Controls the Freenove tank robot over a serial connection."""

    CMD_MOTOR = "CMD_MOTOR"

    def __init__(self, cfg: dict):
        robot_cfg = cfg["robot"]
        self._port = robot_cfg["port"]
        self._baud = robot_cfg["baud"]
        self._timeout = robot_cfg["connection_timeout"]
        self._speed_full = robot_cfg["speed_full"]
        self._speed_slow = robot_cfg["speed_slow"]
        self._reroute_secs = robot_cfg["reroute_turn_seconds"]
        self._reroute_dir = robot_cfg["reroute_direction"]
        self._safe_stop_hold = robot_cfg["safe_stop_hold_seconds"]
        self._ser = None

    def connect(self) -> None:
        import serial  # type: ignore
        self._ser = serial.Serial(
            self._port, self._baud, timeout=self._timeout
        )
        time.sleep(2)  # allow ESP32 to reset after DTR toggle
        logger.info("Connected to Freenove robot on %s", self._port)

    def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            self.stop()
            self._ser.close()
            logger.info("Disconnected from robot")

    # ── High-level movement commands ──────────────────────────────────────────

    def forward(self, speed: int | None = None) -> None:
        s = speed or self._speed_full
        self._send_motor(s, 0, s, 0)

    def slow_forward(self) -> None:
        self._send_motor(self._speed_slow, 0, self._speed_slow, 0)

    def stop(self) -> None:
        self._send_motor(0, 0, 0, 0)

    def reroute(self) -> None:
        """Stop, turn to avoid the obstacle, then resume slow forward."""
        self.stop()
        time.sleep(0.3)
        if self._reroute_dir == "left":
            self._turn_left()
        else:
            self._turn_right()
        time.sleep(self._reroute_secs)
        self.stop()

    def safe_stop(self) -> None:
        """Emergency stop: brake and hold for a defined duration."""
        self.stop()
        logger.warning("SAFE STOP triggered – holding for %.1f s", self._safe_stop_hold)
        time.sleep(self._safe_stop_hold)

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _turn_left(self) -> None:
        # Left motors backward, right motors forward
        self._send_motor(0, self._speed_slow, self._speed_slow, 0)

    def _turn_right(self) -> None:
        # Left motors forward, right motors backward
        self._send_motor(self._speed_slow, 0, 0, self._speed_slow)

    def _send_motor(self, lf: int, lb: int, rf: int, rb: int) -> None:
        cmd = f"{self.CMD_MOTOR}#{lf}#{lb}#{rf}#{rb}#\n"
        if self._ser and self._ser.is_open:
            self._ser.write(cmd.encode("utf-8"))
            logger.debug("Motor cmd: %s", cmd.strip())
        else:
            logger.error("Serial not connected – command dropped: %s", cmd.strip())


class MockRobot:
    """
    Simulated robot for demo / offline mode.

    Logs all commands so the full pipeline can be exercised without hardware.
    """

    def __init__(self, cfg: dict):
        robot_cfg = cfg["robot"]
        self._speed_full = robot_cfg["speed_full"]
        self._speed_slow = robot_cfg["speed_slow"]
        self._reroute_secs = robot_cfg["reroute_turn_seconds"]
        self._reroute_dir = robot_cfg["reroute_direction"]
        self._safe_stop_hold = robot_cfg["safe_stop_hold_seconds"]
        self.last_command = "NONE"

    def connect(self) -> None:
        logger.info("[MockRobot] Connected (simulation)")

    def disconnect(self) -> None:
        logger.info("[MockRobot] Disconnected")

    def forward(self, speed: int | None = None) -> None:
        s = speed or self._speed_full
        self.last_command = f"FORWARD (speed={s})"
        logger.info("[MockRobot] %s", self.last_command)

    def slow_forward(self) -> None:
        self.last_command = f"SLOW_FORWARD (speed={self._speed_slow})"
        logger.info("[MockRobot] %s", self.last_command)

    def stop(self) -> None:
        self.last_command = "STOP"
        logger.info("[MockRobot] %s", self.last_command)

    def reroute(self) -> None:
        self.last_command = f"REROUTE ({self._reroute_dir})"
        logger.info("[MockRobot] %s", self.last_command)

    def safe_stop(self) -> None:
        self.last_command = "SAFE_STOP"
        logger.warning("[MockRobot] SAFE STOP")


def build_robot(cfg: dict):
    """Factory: return real robot in live mode, mock in demo mode."""
    if cfg.get("mode") == "live":
        return RobotController(cfg)
    return MockRobot(cfg)


def execute_action(robot, action: str) -> None:
    """Map a Decision action string to a robot method call."""
    from src.decision import Action

    if action == Action.FORWARD:
        robot.forward()
    elif action == Action.SLOW:
        robot.slow_forward()
    elif action == Action.STOP:
        robot.stop()
    elif action == Action.REROUTE:
        robot.reroute()
    else:
        logger.warning("Unknown action: %s", action)
