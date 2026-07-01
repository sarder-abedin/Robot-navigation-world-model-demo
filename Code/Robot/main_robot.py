"""
main_robot.py – Raspberry Pi robot client entry point.

In this split-inference architecture the robot is the TCP CLIENT:
  - Connects outbound to the PC AI server (ports 5004/8004)
  - Streams JPEG camera frames to the PC
  - Sends ultrasonic readings to the PC
  - Receives CMD_MOTOR / CMD_STOP commands from the PC
  - Executes motor commands via tankMotor

Usage:
  # Live mode (physical robot):
  python main_robot.py --server-ip 192.168.1.42

  # Demo mode (no hardware, for integration testing):
  python main_robot.py --mode demo --server-ip 127.0.0.1
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import yaml

logger = logging.getLogger(__name__)
_shutdown = False


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
        camera = Camera(
            stream_size=(
                cam_cfg.get("stream_width", 400),
                cam_cfg.get("stream_height", 300),
            )
        )
        camera.start_stream()
        motor = tankMotor()
        ultrasonic = Ultrasonic()
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

    # ── Camera streaming thread ──────────────────────────────────────────────
    def camera_loop():
        while not _shutdown and client.is_connected:
            if mode == "live" and camera:
                try:
                    jpg = camera.get_frame()
                    if jpg:
                        client.send_frame(jpg)
                except Exception as exc:
                    logger.warning("Camera error: %s", exc)
                    time.sleep(0.05)
            else:
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

    # ── Ultrasonic broadcast thread ──────────────────────────────────────────
    sonic_interval = cfg.get("ultrasonic", {}).get("read_interval", 0.1)

    def sonic_loop():
        while not _shutdown and client.is_connected:
            if mode == "live" and ultrasonic:
                try:
                    cm = ultrasonic.get_distance()
                    client.send_sonic(cm)
                except Exception as exc:
                    logger.warning("Ultrasonic error: %s", exc)
            else:
                client.send_sonic(-1.0)
            time.sleep(sonic_interval)

    sonic_thread = threading.Thread(target=sonic_loop, daemon=True, name="SonicBroadcast")
    sonic_thread.start()

    # ── Command receive loop (main thread) ───────────────────────────────────
    robot_cfg = cfg.get("robot", {})
    speed_full = robot_cfg.get("speed_full", 1500)
    speed_slow = robot_cfg.get("speed_slow", 800)

    while not _shutdown and client.is_connected:
        cmd = client.get_command(timeout=0.5)
        if cmd is None:
            continue

        parts = cmd.split("#")
        command = parts[0].strip()

        if command == "CMD_MOTOR" and len(parts) >= 3:
            try:
                left = int(parts[1])
                right = int(parts[2])
                if motor:
                    motor.setMotorModel(left, right)
                else:
                    logger.info("[MockMotor] L=%d  R=%d", left, right)
            except (ValueError, IndexError) as exc:
                logger.warning("Bad CMD_MOTOR: %s (%s)", cmd, exc)

        elif command == "CMD_STOP":
            if motor:
                motor.setMotorModel(0, 0)
            logger.info("Motors stopped (CMD_STOP)")

        elif command == "CMD_AIMODE":
            # AI stopped by operator – halt motors for safety
            if motor:
                motor.setMotorModel(0, 0)
            logger.info("AI mode change: %s", cmd)

        elif command == "CMD_KILL":
            logger.warning("CMD_KILL received – robot shutting down")
            _shutdown = True

    # ── Shutdown ─────────────────────────────────────────────────────────────
    _cleanup(motor, camera, ultrasonic, client)


def _cleanup(motor, camera, ultrasonic, client) -> None:
    logger.info("Robot client shutting down…")
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
