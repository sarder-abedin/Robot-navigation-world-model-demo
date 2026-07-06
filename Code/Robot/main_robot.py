"""
main_robot.py – Raspberry Pi robot client entry point (thin client).

All AI runs on the PC server. The Pi is a thin client with no on-board AI, so
its Docker image carries no torch/ultralytics.

Responsibilities
────────────────
  Pi (this machine):
    • Streams JPEG camera frames to PC (port 8004) for ALL server-side AI
      (YOLO11n + V-JEPA 2 + SSv2)
    • Reads the ultrasonic sensor and sends CMD_SONIC (its local hard-stop safety)
    • Executes CMD_AIMOVE (AI-computed actions) and CMD_MOTOR (manual) commands

  PC server (main_server.py):
    • Decodes the camera stream → runs YOLO11n + V-JEPA 2 + SSv2 + decision fusion
    • Reads CMD_SONIC for the ultrasonic hard-stop guard
    • Sends CMD_AIMOVE#<FORWARD|SLOW|STOP|REROUTE> (AI mode)
    • Sends CMD_MOTOR#<L>#<R> (manual mode from operator UI)

Usage:
  # Live mode (physical robot):
  python main_robot.py --server-ip 192.168.1.42

  # Demo mode (no hardware, for integration testing):
  python main_robot.py --mode demo --server-ip 127.0.0.1
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

import yaml

logger = logging.getLogger(__name__)
_shutdown = False

# REROUTE runs as a timed maneuver. It must not block the command loop, or a
# STOP/KILL/MOTOR arriving mid-reroute would be ignored for ~1.5 s. It runs in a
# worker thread that a new command can preempt via this cancel event.
_reroute_thread: threading.Thread | None = None
_reroute_cancel = threading.Event()
_reroute_direction: str = ""   # direction of the in-progress reroute (to detect changes)


def _cancel_reroute() -> None:
    """Stop any in-progress reroute maneuver so a new command takes effect now."""
    global _reroute_thread
    if _reroute_thread is not None and _reroute_thread.is_alive():
        _reroute_cancel.set()
        _reroute_thread.join(timeout=1.0)
    _reroute_thread = None


def _start_reroute(motor, speed_slow: int, reroute_secs: float,
                   direction: str = "left") -> None:
    """Run the back-up + spin maneuver in a preemptible worker thread.

    direction ("left"/"right") comes from the PC depth channel's clear side.
    """
    global _reroute_thread
    _cancel_reroute()
    _reroute_cancel.clear()
    # Tank spin: left = (-L, +R), right = (+L, -R).
    spin_l, spin_r = (-speed_slow, speed_slow) if direction != "right" else (speed_slow, -speed_slow)

    def _run():
        motor.setMotorModel(-speed_slow, -speed_slow)   # back up
        if _reroute_cancel.wait(0.3):                   # preempted?
            motor.setMotorModel(0, 0)
            return
        motor.setMotorModel(spin_l, spin_r)             # spin toward the open side
        if _reroute_cancel.wait(reroute_secs):
            motor.setMotorModel(0, 0)
            return
        motor.setMotorModel(0, 0)

    _reroute_thread = threading.Thread(target=_run, daemon=True, name="Reroute")
    _reroute_thread.start()


def _start_backup(motor, speed_slow: int, secs: float = 0.4) -> None:
    """Short reverse pulse in the preemptible maneuver thread (STOP/KILL can cut it)."""
    global _reroute_thread
    _cancel_reroute()
    _reroute_cancel.clear()

    def _run():
        motor.setMotorModel(-speed_slow, -speed_slow)   # reverse
        if _reroute_cancel.wait(secs):                  # preempted?
            motor.setMotorModel(0, 0)
            return
        motor.setMotorModel(0, 0)

    _reroute_thread = threading.Thread(target=_run, daemon=True, name="Backup")
    _reroute_thread.start()


def _env_bool(name: str, default: bool) -> bool:
    """Read a truthy/falsy env var (1/0/true/false/yes/no); fall back to default."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read an integer env var; fall back to default on unset/invalid."""
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        logger.warning("%s must be an integer, got %r – using %d", name, v, default)
        return default


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def parse_args():
    p = argparse.ArgumentParser(description="Freenove Tank Robot Client")
    p.add_argument("--config", default="config_robot.yaml")
    p.add_argument("--server-ip", default=None, help="Override config: PC server IP")
    p.add_argument("--mode", choices=["live", "demo"], default=None)
    return p.parse_args()


def main() -> None:
    global _shutdown
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mode = args.mode or cfg.get("mode", "live")
    server_ip = args.server_ip or cfg["server_ip"]
    cmd_port = cfg.get("robot_cmd_port", 5004)
    video_port = cfg.get("robot_video_port", 8004)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Hardware initialisation ──────────────────────────────────────────────
    camera = None
    motor = None
    ultrasonic = None

    if mode == "live":
        from camera import Camera
        from motor import tankMotor
        from ultrasonic import Ultrasonic

        cam_cfg = cfg.get("camera", {})
        # Orientation: default from config, but allow -e CAMERA_HFLIP / CAMERA_VFLIP
        # to correct an upside-down feed at runtime without rebuilding the image.
        hflip = _env_bool("CAMERA_HFLIP", cam_cfg.get("hflip", False))
        vflip = _env_bool("CAMERA_VFLIP", cam_cfg.get("vflip", False))
        camera = Camera(
            stream_size=(
                cam_cfg.get("stream_width", 400),
                cam_cfg.get("stream_height", 300),
            ),
            hflip=hflip,
            vflip=vflip,
        )
        try:
            camera.start_stream()
        except Exception as exc:
            # A camera failure must not take down the whole robot – manual driving
            # and the command channel still work without it.
            logger.error("Camera init failed (%s) – continuing without camera stream", exc)
            camera = None

        _gpio_env = os.environ.get("GPIO_CHIP")
        if _gpio_env is not None:
            try:
                gpio_chip = int(_gpio_env)
            except ValueError:
                logger.error(
                    "GPIO_CHIP env var must be an integer (e.g. 0 or 4), got: %r -- exiting",
                    _gpio_env,
                )
                sys.exit(1)
        else:
            gpio_chip = int(cfg.get("gpio", {}).get("chip", 0))
        sonic_cfg = cfg.get("ultrasonic", {})

        try:
            motor = tankMotor(
                gpiochip=gpio_chip,
                soft_start=cfg.get("robot", {}).get("soft_start", True),
            )
        except Exception as exc:
            logger.warning(
                "Motor init failed (%s) – motor disabled. "
                "If this says 'cannot open gpiochip', check --device /dev/gpiochip%d "
                "and set -e GPIO_CHIP=%d in docker run.",
                exc, gpio_chip, gpio_chip,
            )
            motor = None

        try:
            ultrasonic = Ultrasonic(
                trigger_pin=sonic_cfg.get("trigger_pin", 27),
                echo_pin=sonic_cfg.get("echo_pin", 22),
                gpiochip=gpio_chip,
            )
        except Exception as exc:
            logger.warning("Ultrasonic init failed (%s) – sensor disabled", exc)
            ultrasonic = None

        # No YOLO on the Pi. All AI (YOLO11n, V-JEPA 2, SSv2, decision) runs on the
        # PC server; the Pi is a thin client that streams the camera, drives the
        # motors, and reports the ultrasonic distance (its local hard-stop safety).
        logger.info("Hardware initialised in live mode (thin client – no on-Pi AI)")
    else:
        logger.info("Demo mode – stub hardware (no GPIO/camera)")

    # ── TCP connection ───────────────────────────────────────────────────────
    from tcp_robot_client import RobotTCPClient
    client = RobotTCPClient(server_ip, cmd_port, video_port)

    # Motor / command parameters (read once; threads and the command loop below
    # close over these across reconnects).
    robot_cfg = cfg.get("robot", {})
    # Speeds are PWM duty out of 4095. Keep them slow for a reactive demo. Tune
    # at runtime without rebuilding via -e SPEED_FULL=<n> / -e SPEED_SLOW=<n>.
    speed_full = _env_int("SPEED_FULL", robot_cfg.get("speed_full", 1600))
    speed_slow = _env_int("SPEED_SLOW", robot_cfg.get("speed_slow", 1000))
    reroute_secs = robot_cfg.get("reroute_turn_seconds", 1.2)
    watchdog_timeout = float(robot_cfg.get("command_watchdog_seconds", 1.5))
    sonic_interval = cfg.get("ultrasonic", {}).get("read_interval", 0.1)
    sonic_cfg = cfg.get("ultrasonic", {})
    logger.info("Motor speeds: FORWARD=%d  SLOW=%d  (of 4095 max)", speed_full, speed_slow)

    # ── Camera streaming thread ──────────────────────────────────────────────
    # ONLY camera_loop calls camera.get_frame(); the PC decodes the stream and
    # runs all detection, so the Pi never needs a second capture.
    def camera_loop():
        """Continuously stream JPEG frames to the PC for server-side AI."""
        frames_sent = 0
        none_ticks = 0
        while not _shutdown and client.is_connected:
            if mode == "live" and camera:
                try:
                    jpg = camera.get_frame()
                    if jpg:
                        client.send_frame(jpg)
                        frames_sent += 1
                        if frames_sent == 1:
                            logger.info("Camera streaming to PC – first frame sent (%d bytes)", len(jpg))
                        none_ticks = 0
                    else:
                        none_ticks += 1
                        # Warn ~once/5s if the camera keeps returning no frame
                        if none_ticks % 150 == 1:
                            logger.warning(
                                "Camera get_frame() returned no frame – camera opened but "
                                "not producing images (check CSI ribbon / /dev/video0)."
                            )
                        time.sleep(0.03)
                except Exception as exc:
                    logger.warning("Camera stream error: %s", exc)
                    time.sleep(0.05)
            elif mode == "live":
                # Live mode but the camera failed to initialise (camera is None).
                # Do NOT stream a fake black frame: V-JEPA 2 reads it as an
                # obstacle ("BLOCKED"), so the robot lurches forward then stops.
                # Idle instead — the PC reports "waiting for camera frames" and
                # the robot stays put until the camera is fixed (see the
                # "Camera init failed" error logged at startup).
                none_ticks += 1
                if none_ticks % 100 == 1:
                    logger.warning(
                        "Live mode but camera is unavailable – not streaming video. "
                        "Fix the camera (see 'Camera init failed' at startup; usually "
                        "a missing '-v /run/udev:/run/udev:ro' mount) and restart."
                    )
                time.sleep(0.1)
            else:
                # Demo mode (no hardware) – send a labelled placeholder frame.
                import cv2
                import numpy as np
                blank = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(blank, "DEMO ROBOT", (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
                _, buf = cv2.imencode(".jpg", blank)
                client.send_frame(buf.tobytes())
                time.sleep(0.1)

    # ── Ultrasonic broadcast thread ──────────────────────────────────────────
    def sonic_loop():
        """
        Reads the ultrasonic sensor and sends a CMD_SONIC message per cycle to
        the PC. This is the Pi's only sensor report — object detection runs on
        the PC from the streamed camera frames.
        """
        sonic_dead_ticks = 0   # consecutive max/error readings → sensor likely down

        while not _shutdown and client.is_connected:
            sonic_cm = -1.0
            if mode == "live" and ultrasonic:
                try:
                    sonic_cm = ultrasonic.get_distance()
                except Exception as exc:
                    logger.warning("Ultrasonic error: %s", exc)
                # A sensor that never echoes pins at max range (≈400 cm), so the
                # hard STOP guard never fires — warn loudly, it's a safety issue.
                if sonic_cm < 0 or sonic_cm >= 395:
                    sonic_dead_ticks += 1
                    if sonic_dead_ticks % 100 == 1:
                        logger.warning(
                            "Ultrasonic reads no echo (%.0f cm) – the distance safety "
                            "STOP is effectively DISABLED. Check the HC-SR04 wiring "
                            "(trigger=%s echo=%s) / 5V power / echo level-shift.",
                            sonic_cm, sonic_cfg.get("trigger_pin", 27),
                            sonic_cfg.get("echo_pin", 22),
                        )
                else:
                    sonic_dead_ticks = 0

            client.send_sonic(sonic_cm)
            time.sleep(sonic_interval)

    # ── Connect + run, with an outer reconnect loop ──────────────────────────
    # A transient network blip (Wi-Fi hiccup, PC restart) should drop the robot
    # back to (re)connecting — with the motors stopped — instead of killing the
    # client permanently. Each successful connection spins up fresh streaming
    # threads and runs the command loop until the link drops or we shut down.
    while not _shutdown:
        logger.info("Connecting to PC server at %s (cmd=%d, video=%d)…",
                    server_ip, cmd_port, video_port)
        connected = False
        while not _shutdown:
            if client.connect(timeout=5.0):
                connected = True
                break
            logger.warning("Connection failed – retrying in 2 s…")
            time.sleep(2.0)
        if not connected:
            break   # _shutdown requested while (re)connecting

        logger.info("Connected to PC server – robot client running")
        cam_thread = threading.Thread(target=camera_loop, daemon=True, name="CameraStream")
        cam_thread.start()
        det_thread = threading.Thread(target=sonic_loop, daemon=True, name="SonicBroadcast")
        det_thread.start()

        _command_loop(client, motor, speed_full, speed_slow, reroute_secs, watchdog_timeout)

        # Command loop returned: link dropped or shutdown requested. Fail safe —
        # stop the motors and any maneuver — before winding threads down.
        _cancel_reroute()
        if motor:
            motor.setMotorModel(0, 0)
        cam_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        if not _shutdown:
            logger.warning("Lost connection to PC – reconnecting…")
            client.disconnect()   # reset socket state before the next connect()
            time.sleep(1.0)

    # ── Shutdown ─────────────────────────────────────────────────────────────
    _cleanup(motor, camera, ultrasonic, client)


def _command_loop(client, motor, speed_full: int, speed_slow: int,
                  reroute_secs: float, watchdog_timeout: float) -> None:
    """Run the PC→Pi command loop until the link drops or shutdown is requested.

    Motor watchdog (failsafe): if no command arrives from the PC within
    watchdog_timeout seconds — a stalled pipeline/server, a stalled video
    stream, or a silently half-open TCP link — the last drive command would
    otherwise keep the motors running into an obstacle. Stop them until traffic
    resumes. TCP keepalive (see tcp_robot_client) eventually tears down a truly
    dead socket so the loop exits and the outer loop reconnects.
    """
    global _shutdown
    last_cmd_time = time.monotonic()
    watchdog_tripped = False

    while not _shutdown and client.is_connected:
        cmd = client.get_command(timeout=0.5)
        if cmd is None:
            if (watchdog_timeout > 0 and not watchdog_tripped
                    and (time.monotonic() - last_cmd_time) > watchdog_timeout):
                _cancel_reroute()
                if motor:
                    motor.setMotorModel(0, 0)
                watchdog_tripped = True
                logger.warning(
                    "No command from PC for %.1fs – motor watchdog STOP (failsafe).",
                    watchdog_timeout,
                )
            continue

        # Any command means the PC is alive and driving us again.
        last_cmd_time = time.monotonic()
        watchdog_tripped = False

        parts = cmd.split("#")
        command = parts[0].strip()

        if command == "CMD_AIMOVE" and len(parts) >= 2:
            # AI-computed navigation action from the PC decision fuser.
            # REROUTE may carry a turn direction: CMD_AIMOVE#REROUTE#LEFT|RIGHT
            action = parts[1].strip()
            direction = parts[2].strip().lower() if len(parts) >= 3 else ""
            _execute_aimove(action, motor, speed_full, speed_slow, reroute_secs, direction)

        elif command == "CMD_MOTOR" and len(parts) >= 3:
            # Manual motor command from the operator UI (relayed by PC)
            _cancel_reroute()
            try:
                left = int(parts[1])
                right = int(parts[2])
                if motor:
                    motor.setMotorModel(left, right)
                else:
                    logger.info("[MockMotor] CMD_MOTOR L=%d R=%d", left, right)
            except (ValueError, IndexError) as exc:
                logger.warning("Bad CMD_MOTOR: %s (%s)", cmd, exc)

        elif command == "CMD_STOP":
            _cancel_reroute()
            if motor:
                motor.setMotorModel(0, 0)
            logger.info("Motors stopped (CMD_STOP)")

        elif command == "CMD_AIMODE":
            # AI mode change – stop motors for safety on mode transition
            _cancel_reroute()
            if motor:
                motor.setMotorModel(0, 0)
            logger.info("AI mode change: %s", cmd)

        elif command == "CMD_KILL":
            logger.warning("CMD_KILL received – robot shutting down")
            _cancel_reroute()
            _shutdown = True


def _execute_aimove(action: str, motor, speed_full: int, speed_slow: int,
                    reroute_secs: float, direction: str = "") -> None:
    """Map an AI navigation action string to tankMotor calls."""
    # Any new action preempts an in-progress (non-blocking) reroute maneuver.
    if action != "REROUTE":
        _cancel_reroute()

    if action == "FORWARD":
        if motor:
            motor.setMotorModel(speed_full, speed_full)
        else:
            logger.info("[MockMotor] FORWARD L=%d R=%d", speed_full, speed_full)

    elif action == "SLOW":
        if motor:
            motor.setMotorModel(speed_slow, speed_slow)
        else:
            logger.info("[MockMotor] SLOW L=%d R=%d", speed_slow, speed_slow)

    elif action == "STOP":
        if motor:
            motor.setMotorModel(0, 0)
        else:
            logger.info("[MockMotor] STOP")

    elif action == "BACKUP":
        # Short reverse pulse (PC decided the obstacle is too close to turn). Runs
        # in the preemptible maneuver thread so a STOP/KILL isn't blocked by it.
        if motor:
            _start_backup(motor, speed_slow)
        else:
            logger.info("[MockMotor] BACKUP")

    elif action == "REROUTE":
        # Timed back-up + spin runs in a worker thread so STOP/KILL/MOTOR can
        # preempt it instead of being ignored for the full maneuver. The turn
        # direction comes from the PC depth channel (open side); default left.
        # The PC re-sends REROUTE every frame while it wants us to keep turning;
        # restarting the maneuver each frame would trap the robot in the initial
        # back-up phase and it would never actually spin (forward/backward
        # oscillation). So let an in-progress reroute in the SAME direction run to
        # completion — only (re)start on a fresh reroute or a direction change.
        global _reroute_direction
        turn = direction if direction in ("left", "right") else "left"
        if motor:
            running = _reroute_thread is not None and _reroute_thread.is_alive()
            if not running or turn != _reroute_direction:
                _reroute_direction = turn
                _start_reroute(motor, speed_slow, reroute_secs, turn)
        else:
            logger.info("[MockMotor] REROUTE %s (%.1f s)", turn, reroute_secs)

    else:
        logger.warning("Unknown CMD_AIMOVE action: %s", action)


def _cleanup(motor, camera, ultrasonic, client) -> None:
    logger.info("Robot client shutting down…")
    _cancel_reroute()
    try:
        if motor:
            motor.setMotorModel(0, 0)
            motor.close()
        if camera:
            camera.close()
        if ultrasonic:
            ultrasonic.close()
        if client:
            client.disconnect()
    except Exception as exc:
        logger.error("Cleanup error: %s", exc)
    logger.info("Robot client stopped")


if __name__ == "__main__":
    main()
