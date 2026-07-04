"""
main_robot.py – Raspberry Pi robot client entry point (split-inference arch).

Split-inference responsibilities
─────────────────────────────────
  Pi (this machine):
    • Streams JPEG camera frames to PC (port 8004) for V-JEPA 2 inference
    • Runs YOLOv8n locally, sends CMD_DETECTION (fused detection + sonic) to PC
    • Executes CMD_AIMOVE (AI-computed actions) and CMD_MOTOR (manual) commands
    • Reads the ultrasonic sensor; distance is included in CMD_DETECTION

  PC server (main_server.py):
    • Receives CMD_DETECTION → runs V-JEPA 2 + SSv2 + decision fusion
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


def _cancel_reroute() -> None:
    """Stop any in-progress reroute maneuver so a new command takes effect now."""
    global _reroute_thread
    if _reroute_thread is not None and _reroute_thread.is_alive():
        _reroute_cancel.set()
        _reroute_thread.join(timeout=1.0)
    _reroute_thread = None


def _start_reroute(motor, speed_slow: int, reroute_secs: float) -> None:
    """Run the back-up + spin maneuver in a preemptible worker thread."""
    global _reroute_thread
    _cancel_reroute()
    _reroute_cancel.clear()

    def _run():
        motor.setMotorModel(-speed_slow, -speed_slow)   # back up
        if _reroute_cancel.wait(0.3):                   # preempted?
            motor.setMotorModel(0, 0)
            return
        motor.setMotorModel(-speed_slow, speed_slow)    # spin left
        if _reroute_cancel.wait(reroute_secs):
            motor.setMotorModel(0, 0)
            return
        motor.setMotorModel(0, 0)

    _reroute_thread = threading.Thread(target=_run, daemon=True, name="Reroute")
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
    detector = None

    if mode == "live":
        from camera import Camera
        from motor import tankMotor
        from ultrasonic import Ultrasonic
        from detector_robot import DetectorRobot

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

        # Detection location: run YOLO on the Pi ("pi") or offload it to the PC
        # ("server"). In server mode the Pi skips YOLOv8n entirely (big CPU/current
        # saving – fixes battery brownouts) and just streams frames + ultrasonic.
        det_location = os.environ.get(
            "DETECTOR_LOCATION", str(cfg.get("detector", {}).get("location", "pi"))
        ).strip().lower()
        if det_location == "server":
            detector = None
            logger.info(
                "DETECTOR_LOCATION=server – YOLO runs on the PC. Pi skips YOLOv8n "
                "to save power (still streams camera + reads ultrasonic)."
            )
        else:
            detector = DetectorRobot(cfg)
            try:
                detector.load()
                logger.info("YOLOv8n loaded on Pi")
            except Exception as exc:
                logger.warning("YOLOv8n load failed (%s) – detection disabled", exc)
                detector = None

        logger.info("Hardware initialised in live mode")
    else:
        logger.info("Demo mode – stub hardware (no GPIO/camera)")

    # ── TCP connection ───────────────────────────────────────────────────────
    from tcp_robot_client import RobotTCPClient
    client = RobotTCPClient(server_ip, cmd_port, video_port)

    logger.info("Connecting to PC server at %s (cmd=%d, video=%d)…",
                server_ip, cmd_port, video_port)
    while not _shutdown:
        if client.connect(timeout=5.0):
            break
        logger.warning("Connection failed – retrying in 2 s…")
        time.sleep(2.0)

    if _shutdown:
        _cleanup(motor, camera, ultrasonic, client)
        return

    logger.info("Connected to PC server – robot client running")

    # Single capture point: ONLY camera_loop calls camera.get_frame(). The
    # detection loop reads this cache instead of capturing again — two threads
    # calling picamera2 capture_array() concurrently starves the (Docker-limited)
    # buffer pool and deadlocks on the camera lock, hanging after the first frame.
    _frame_cache = {"jpg": None}
    _frame_cache_lock = threading.Lock()

    # ── Camera streaming thread ──────────────────────────────────────────────
    def camera_loop():
        """Continuously stream JPEG frames to the PC for V-JEPA 2 inference."""
        frames_sent = 0
        none_ticks = 0
        while not _shutdown and client.is_connected:
            if mode == "live" and camera:
                try:
                    jpg = camera.get_frame()
                    if jpg:
                        with _frame_cache_lock:
                            _frame_cache["jpg"] = jpg
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

    cam_thread = threading.Thread(target=camera_loop, daemon=True, name="CameraStream")
    cam_thread.start()

    # ── Detection + sonic broadcast thread ──────────────────────────────────
    sonic_interval = cfg.get("ultrasonic", {}).get("read_interval", 0.1)

    def detection_loop():
        """
        Runs YOLOv8n on the latest camera frame and reads the ultrasonic sensor,
        then sends a single CMD_DETECTION message per cycle to the PC.
        """
        from detector_robot import DetectionPacket
        last_packet = DetectionPacket()
        sonic_dead_ticks = 0   # consecutive max/error readings → sensor likely down

        while not _shutdown and client.is_connected:
            # ── 1. YOLOv8n on the latest cached frame (live mode only) ────────
            # Read the frame camera_loop already captured — do NOT call
            # camera.get_frame() here (a second capture_array() would deadlock the
            # camera buffer pool).
            if mode == "live" and detector:
                try:
                    with _frame_cache_lock:
                        jpg = _frame_cache["jpg"]
                    if jpg:
                        import cv2
                        import numpy as np
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            last_packet = detector.detect(frame_bgr)
                except Exception as exc:
                    logger.warning("YOLO detection error: %s", exc)

            # ── 2. Ultrasonic distance ────────────────────────────────────────
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

            # ── 3. Send fused CMD_DETECTION to PC ────────────────────────────
            client.send_detection(
                risk_pct=last_packet.yolo_risk_pct,
                obs_in_center=last_packet.obs_in_center,
                area_frac_pct=last_packet.area_frac_pct,
                centroid_x_pct=last_packet.centroid_x_pct,
                sonic_cm=sonic_cm,
                top_label=last_packet.top_label,
            )
            time.sleep(sonic_interval)

    det_thread = threading.Thread(target=detection_loop, daemon=True, name="DetectionBroadcast")
    det_thread.start()

    # ── Command receive loop (main thread) ───────────────────────────────────
    robot_cfg = cfg.get("robot", {})
    # Speeds are PWM duty out of 4095. Keep them slow for a reactive demo. Tune
    # at runtime without rebuilding via -e SPEED_FULL=<n> / -e SPEED_SLOW=<n>.
    speed_full = _env_int("SPEED_FULL", robot_cfg.get("speed_full", 1600))
    speed_slow = _env_int("SPEED_SLOW", robot_cfg.get("speed_slow", 1000))
    reroute_secs = robot_cfg.get("reroute_turn_seconds", 1.2)
    logger.info("Motor speeds: FORWARD=%d  SLOW=%d  (of 4095 max)", speed_full, speed_slow)

    while not _shutdown and client.is_connected:
        cmd = client.get_command(timeout=0.5)
        if cmd is None:
            continue

        parts = cmd.split("#")
        command = parts[0].strip()

        if command == "CMD_AIMOVE" and len(parts) >= 2:
            # AI-computed navigation action from the PC decision fuser
            action = parts[1].strip()
            _execute_aimove(action, motor, speed_full, speed_slow, reroute_secs)

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

    # ── Shutdown ─────────────────────────────────────────────────────────────
    _cleanup(motor, camera, ultrasonic, client)


def _execute_aimove(action: str, motor, speed_full: int, speed_slow: int,
                    reroute_secs: float) -> None:
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

    elif action == "REROUTE":
        # Timed back-up + spin runs in a worker thread so STOP/KILL/MOTOR can
        # preempt it instead of being ignored for the full maneuver.
        if motor:
            _start_reroute(motor, speed_slow, reroute_secs)
        else:
            logger.info("[MockMotor] REROUTE (%.1f s)", reroute_secs)

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
