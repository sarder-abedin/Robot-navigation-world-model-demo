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


def _distance_to_risk(
    d_cm: float,
    stop_cm: float,
    state: dict,
    blind_hold_s: float = 1.0,
) -> float:
    """Map an ultrasonic distance (cm) to a hard-stop risk in [0,1].

    This is the deterministic safety layer (see decision.py): it reaches 1.0 only
    when an obstacle is within stop_cm, which triggers the hard STOP. It is NOT
    blended into the AI risk — vision handles graded slowing/rerouting.

    The driver returns -1.0 (<= 0) when the sensor got no echo this cycle. A
    single missed ping must not drop a close obstacle back to "clear", so we
    remember the last valid reading and reuse it for blind_hold_s seconds. If the
    sensor stays blind longer than that we return 0.0 (no hard stop) and let
    vision drive — the last-good hold still catches a wall the sensor briefly saw.

    `state` is a per-controller dict {"cm": float|None, "t": monotonic_seconds}.
    """
    now = time.monotonic()
    if d_cm is not None and d_cm > 0:
        state["cm"] = d_cm
        state["t"] = now
    else:
        if state.get("t") and state.get("cm") and (now - state["t"]) <= blind_hold_s:
            d_cm = state["cm"]          # reuse a very recent valid reading
        else:
            return 0.0                  # blind too long → no hard stop; vision drives
    if d_cm <= stop_cm:
        return 1.0
    warn = stop_cm * 3.0
    if d_cm >= warn:
        return 0.0
    return float((warn - d_cm) / (warn - stop_cm))


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
        self._blind_hold_s = r.get("ultrasonic_blind_hold_seconds", 1.0)
        self._sonic_state = {"cm": None, "t": 0.0}
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

    def reroute(self, direction: str = "") -> None:
        """Stop, turn to avoid obstacle, then resume slow forward.

        direction ("left"/"right", from the depth channel) overrides the
        configured default so the robot turns toward the open side.
        """
        turn = (direction or "").strip().lower() or self._reroute_dir
        self.stop()
        time.sleep(0.2)
        if turn == "left":
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
        """Risk in [0,1] from the ultrasonic distance, failing safe on no-echo."""
        if not self._use_sonic_guard or self._car is None:
            return 0.0
        try:
            d = self._car.sonic.get_distance()
        except Exception as exc:
            logger.debug("Ultrasonic read error: %s", exc)
            d = -1.0
        return _distance_to_risk(
            d, self._sonic_stop_cm, self._sonic_state, self._blind_hold_s,
        )

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

    def reroute(self, direction: str = "") -> None:
        turn = (direction or "").strip().lower() or self._reroute_dir
        self.last_command = f"REROUTE({turn})"
        logger.info("[MockRobot] %s", self.last_command)

    def safe_stop(self) -> None:
        self.last_command = "SAFE_STOP"
        logger.warning("[MockRobot] SAFE_STOP")

    def get_ultrasonic_risk(self) -> float:
        return 0.0


class TCPRobotController:
    """
    Sends AI navigation actions to the robot via RobotConnectionServer.

    In split-inference mode the PC computes the action (FORWARD/SLOW/STOP/REROUTE)
    and sends it as CMD_AIMOVE.  The Pi executes the actual motor PWM mapping
    locally, keeping real-time timing (e.g. reroute manoeuvres) on the hardware.

    Manual CMD_MOTOR commands from the UI are relayed separately by main_server.py
    and do NOT go through this controller.
    """

    def __init__(self, cfg: dict, robot_conn):
        r = cfg["robot"]
        self._sonic_stop_cm = r["ultrasonic_stop_cm"]
        self._use_sonic_guard = r.get("use_ultrasonic_guard", True)
        self._blind_hold_s = r.get("ultrasonic_blind_hold_seconds", 1.0)
        self._sonic_state = {"cm": None, "t": 0.0}
        self._conn = robot_conn
        self._last_sent: str | None = None

    def _send(self, action: str) -> None:
        """Send a CMD_AIMOVE and make the outcome visible in the log.

        send_aimove() returns False when the robot COMMAND channel (port 5004)
        is not connected — which can happen while the VIDEO channel (8004) is
        still delivering frames, so the UI shows live video but the robot never
        moves. Log on every action change, and warn loudly when a command could
        not be delivered so this failure mode is obvious.
        """
        ok = self._conn.send_aimove(action)
        if action != self._last_sent:
            if ok:
                logger.info("AI → robot: CMD_AIMOVE#%s", action)
            else:
                logger.warning(
                    "AI → robot: CMD_AIMOVE#%s NOT delivered – robot command "
                    "channel (port 5004) is not connected", action,
                )
            self._last_sent = action

    def forward(self) -> None:
        if self._sonic_blocked():
            logger.info("Sonic guard: forward blocked – sending STOP")
            self.stop()
            return
        self._send("FORWARD")

    def slow_forward(self) -> None:
        if self._sonic_blocked():
            self.stop()
            return
        self._send("SLOW")

    def stop(self) -> None:
        self._send("STOP")

    def reroute(self, direction: str = "") -> None:
        # Timed manoeuvre executes on the Pi; append the turn direction (from the
        # depth channel) so it turns toward the open side. Empty → Pi default.
        d = (direction or "").strip().lower()
        self._send(f"REROUTE#{d.upper()}" if d in ("left", "right") else "REROUTE")

    def safe_stop(self) -> None:
        self._conn.send_stop()
        logger.warning("SAFE STOP – CMD_STOP sent to robot")

    def get_ultrasonic_risk(self) -> float:
        if not self._use_sonic_guard:
            return 0.0
        try:
            d = self._conn.get_sonic_cm()
        except Exception as exc:
            logger.debug("TCP sonic error: %s", exc)
            d = -1.0
        return _distance_to_risk(
            d, self._sonic_stop_cm, self._sonic_state, self._blind_hold_s,
        )

    def _sonic_blocked(self) -> bool:
        return self._use_sonic_guard and self.get_ultrasonic_risk() >= 1.0


def build_robot_controller(cfg: dict, car=None, robot_conn=None):
    """
    Factory – returns the appropriate robot controller:
      tcp mode:  TCPRobotController  (PC sends motor commands over network)
      live mode: RobotController     (direct gpiozero / car.py interface)
      demo mode: MockRobotController (no hardware)
    """
    if robot_conn is not None:
        return TCPRobotController(cfg, robot_conn)
    if cfg.get("mode") == "live":
        if car is None:
            raise ValueError("car object required in live mode without robot_conn")
        return RobotController(cfg, car)
    return MockRobotController(cfg)


def execute_action(controller, action: str, reroute_direction: str = "") -> None:
    """Map a Decision Action enum value to a controller method.

    reroute_direction ("left"/"right", from the depth free-space channel) tells
    REROUTE which way to turn toward the open side.
    """
    from decision import Action
    if action == Action.REROUTE:
        controller.reroute(reroute_direction)
        return
    dispatch = {
        Action.FORWARD:  controller.forward,
        Action.SLOW:     controller.slow_forward,
        Action.STOP:     controller.stop,
    }
    fn = dispatch.get(action)
    if fn:
        fn()
    else:
        logger.warning("Unknown action: %s", action)
