"""
main_server.py – PC/Laptop server entry point (split-inference architecture).

In this architecture the PC is the SERVER and the Raspberry Pi robot is the CLIENT:

  PC (this machine):
    • Runs all AI  (YOLOv8 + V-JEPA 2 + SSv2 temporal + decision fusion)
    • Listens on ports 5003/8003  for the operator UI viewer (ai_viewer.py)
    • Listens on ports 5004/8004  for the robot (Pi) to connect
    • Shows the OpenCV HUD overlay window (optional)

  Raspberry Pi (Code/Robot/main_robot.py):
    • Connects outbound to this PC on ports 5004/8004
    • Streams camera JPEG frames → port 8004
    • Receives CMD_MOTOR commands ← port 5004
    • Reads ultrasonic, sends CMD_SONIC → port 5004

Usage:
  # Demo mode (no robot hardware needed – uses video file):
  python main_server.py --mode demo --nav predictive
  python main_server.py --mode demo --nav baseline

  # Live mode (robot must connect on 5004/8004):
  python main_server.py --mode live --nav predictive

  # Build V-JEPA 2 anchors for your corridor:
  python main_server.py --build-anchors
"""
from __future__ import annotations

import argparse
import logging
import signal
import struct
import sys
import threading
import time

import yaml

logger = logging.getLogger(__name__)
_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Predictive Navigation PC Server")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mode", choices=["demo", "live"], default=None,
                   help="Override config: 'demo' (video file) or 'live' (robot TCP)")
    p.add_argument("--nav", choices=["predictive", "baseline"], default=None)
    p.add_argument("--video", default=None, help="Demo video path override")
    p.add_argument("--no-display", action="store_true",
                   help="Disable OpenCV HUD window")
    p.add_argument("--logging", choices=["on", "off"], default=None,
                   help="Initial run-logging state (CSV + annotated frames). "
                        "Also settable via NAV_LOGGING=1/0; toggled live from the UI.")
    p.add_argument("--build-anchors", action="store_true",
                   help="Calibrate V-JEPA 2 anchors interactively then exit")
    return p.parse_args()


def _resolve_logging_enabled(args, cfg) -> bool:
    """Resolve the initial logging state: --logging flag > NAV_LOGGING env > config."""
    import os
    if args.logging is not None:
        return args.logging == "on"
    env = os.environ.get("NAV_LOGGING")
    if env is not None:
        return env.strip().lower() in ("1", "true", "on", "yes")
    return bool(cfg.get("logging", {}).get("enabled", False))


# ── PC Navigation Server ───────────────────────────────────────────────────────

class PCNavigationServer:
    """
    PC server that:
      1. Optionally waits for the robot to connect (live mode)
      2. Runs the full AI pipeline in a background thread
      3. Serves the UI viewer on ports 5003/8003 (same protocol as before)
    """

    CMD_AIMODE  = "CMD_AIMODE"
    CMD_KILL    = "CMD_KILL"
    CMD_MOTOR   = "CMD_MOTOR"
    CMD_LOGGING = "CMD_LOGGING"

    def __init__(self, cfg: dict, nav_mode: str):
        self._cfg = cfg
        self._nav_mode = nav_mode

        from camera_buffer import CameraBuffer
        from server import TankServer
        from command import Command
        from message import MessageParser
        from ai_pipeline import AIPipeline

        mode = cfg.get("mode", "demo")

        # Camera buffer – in live mode frames are pushed by robot_connection
        if mode == "live":
            live_cfg = dict(cfg)
            live_cfg["mode"] = "tcp"  # no capture thread; frames come from robot
            self._cam_buf = CameraBuffer(live_cfg)
        else:
            self._cam_buf = CameraBuffer(cfg)

        # Robot TCP connection (live mode only)
        self._robot_conn = None
        if mode == "live":
            from robot_connection import RobotConnectionServer
            self._robot_conn = RobotConnectionServer(cfg, self._cam_buf)

        # AI pipeline
        self._ai = AIPipeline(cfg=self._cfg)

        # UI TCP server (ports 5003/8003 – same as original Freenove protocol)
        self._tcp = TankServer()
        self._command = Command()
        self._parser = MessageParser()

        self._cmd_running = False
        self._video_running = False
        self._cmd_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None

    def start(self) -> None:
        mode = self._cfg.get("mode", "demo")
        srv_cfg = self._cfg.get("server", {})

        # Start robot TCP listener (live mode)
        if mode == "live" and self._robot_conn:
            self._robot_conn.start()

        # Start the UI TCP server (viewer connects here)
        self._tcp.startTcpServer(
            port1=srv_cfg.get("cmd_port", 5003),
            port2=srv_cfg.get("video_port", 8003),
        )

        # Attach everything to the AI pipeline and start it
        self._ai.attach(
            tcp_server=self._tcp,
            robot_connection=self._robot_conn,
            camera_buffer=self._cam_buf,
        )
        self._cam_buf.start()
        self._ai.start()

        # Start the command receive thread (for UI viewer commands)
        self._cmd_running = True
        self._cmd_thread = threading.Thread(
            target=self._cmd_loop, daemon=True, name="UICmdReceive"
        )
        self._cmd_thread.start()

        # Start the video send thread (annotated frames → UI viewer)
        self._video_running = True
        self._video_thread = threading.Thread(
            target=self._video_loop, daemon=True, name="UIVideoSend"
        )
        self._video_thread.start()

        logger.info(
            "PC server started – mode=%s  nav=%s  UI cmd:%d  UI video:%d",
            mode, self._nav_mode,
            srv_cfg.get("cmd_port", 5003),
            srv_cfg.get("video_port", 8003),
        )
        if mode == "live":
            robot_srv_cfg = self._cfg.get("server", {})
            logger.info(
                "Waiting for robot on robot_cmd:%d  robot_video:%d",
                robot_srv_cfg.get("robot_cmd_port", 5004),
                robot_srv_cfg.get("robot_video_port", 8004),
            )

    def stop(self) -> None:
        self._cmd_running = False
        self._video_running = False
        self._ai.stop()
        self._cam_buf.stop()
        self._tcp.stopTcpServer()
        if self._robot_conn:
            self._robot_conn.send_stop()
            self._robot_conn.stop()
        logger.info("PC Navigation Server stopped")

    # ── UI command receive loop ────────────────────────────────────────────────

    def _cmd_loop(self) -> None:
        while self._cmd_running:
            q = self._tcp.readDataFromCmdServer()
            while q.qsize() > 0:
                _, raw = q.get()
                for msg in raw.strip().split("\n"):
                    msg = msg.strip()
                    if msg:
                        self._handle_command(msg)
            time.sleep(0.001)

    def _handle_command(self, msg: str) -> None:
        self._parser.clearParameters()
        self._parser.parser(msg)
        cmd = self._parser.commandString

        if cmd == self.CMD_AIMODE:
            val = self._parser.intParameter[0] if self._parser.intParameter else -1
            if val == 0:
                self._ai.set_motor_enabled(False)
                if self._robot_conn:
                    self._robot_conn.send_stop()
                logger.info("AI control disabled by UI viewer")
            elif val == 1:
                self._ai.set_motor_enabled(True)
                self._ai.set_navigation_mode("baseline")
            elif val == 2:
                self._ai.set_motor_enabled(True)
                self._ai.set_navigation_mode("predictive")
            # Forward mode change to robot only for valid modes (never CMD_AIMODE#-1)
            if self._robot_conn and val in (0, 1, 2):
                self._robot_conn.send_aimode(val)

        elif cmd == self.CMD_MOTOR:
            # Manual motor command from operator UI – disable AI then relay so the
            # pipeline cannot override this command on the next frame.
            if self._robot_conn and len(self._parser.intParameter) >= 2:
                L, R = self._parser.intParameter[0], self._parser.intParameter[1]
                self._ai.set_motor_enabled(False)
                self._robot_conn.send_motor_command(L, R)
                logger.debug("Manual CMD_MOTOR relayed: L=%d R=%d", L, R)
            elif self._robot_conn:
                logger.warning("CMD_MOTOR with unexpected format: %s", msg)

        elif cmd == self.CMD_LOGGING:
            # Operator toggled run logging (CSV + annotated frames) from the UI.
            val = self._parser.intParameter[0] if self._parser.intParameter else 0
            self._ai.set_logging_enabled(val == 1)

        elif cmd == self.CMD_KILL:
            logger.warning("CMD_KILL from UI viewer – shutting down")
            if self._robot_conn:
                self._robot_conn.send_kill()
            self._ai.stop()
            global _shutdown
            _shutdown = True

    # ── UI video send loop ─────────────────────────────────────────────────────

    def _video_loop(self) -> None:
        while self._video_running:
            if not self._tcp.isVideoServerConnected():
                time.sleep(0.1)
                continue
            try:
                jpg = self._ai.get_annotated_jpeg()
                if jpg is None:
                    time.sleep(0.05)
                    continue
                # Send the length prefix and JPEG as ONE buffer so a client can
                # never receive a header without its body (frame desync) if it
                # connects or drops between two separate sends.
                self._tcp.sendDataToVideoClient(struct.pack("<I", len(jpg)) + jpg)
            except Exception as exc:
                logger.debug("Video send error: %s", exc)
            time.sleep(1.0 / 20)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global args
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode:
        cfg["mode"] = args.mode
    if args.video:
        cfg["camera"]["demo_video_path"] = args.video
    if args.no_display:
        cfg["visualization"]["show_window"] = False
    if args.nav:
        cfg["navigation_mode"] = args.nav
    # Initial run-logging state: --logging flag > NAV_LOGGING env > config.
    cfg.setdefault("logging", {})["enabled"] = _resolve_logging_enabled(args, cfg)

    nav_mode = args.nav or cfg.get("navigation_mode", "predictive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.build_anchors:
        _run_anchor_builder(cfg)
        return

    server = PCNavigationServer(cfg, nav_mode)
    server.start()

    mode = cfg.get("mode", "demo")
    srv_cfg = cfg.get("server", {})
    print(f"\n{'='*60}")
    print(f"  Freenove Predictive Navigation – PC Server")
    print(f"  Mode: {mode}  |  Navigation: {nav_mode}")
    print(f"  UI viewer ports:  cmd={srv_cfg.get('cmd_port', 5003)}"
          f"  video={srv_cfg.get('video_port', 8003)}")
    if mode == "live":
        print(f"  Robot ports:      cmd={srv_cfg.get('robot_cmd_port', 5004)}"
              f"  video={srv_cfg.get('robot_video_port', 8004)}")
        print(f"  Start robot:  cd Code/Robot && python main_robot.py"
              f" --server-ip <THIS PC IP>")
    print(f"  Start viewer: cd Code/Client && python ai_viewer.py")
    print(f"                (enter this PC's IP in the viewer)")
    print(f"  Press Ctrl-C to stop")
    print(f"{'='*60}\n")

    try:
        while not _shutdown:
            time.sleep(0.5)
            state = server._ai.get_state()
            if not state.running:
                break
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
    finally:
        server.stop()
        logger.info("PC server shut down cleanly")


def _run_anchor_builder(cfg: dict) -> None:
    import cv2
    from camera_buffer import CameraBuffer
    from world_model import WorldModel

    cam_buf = CameraBuffer(cfg)
    cam_buf.start()
    wm = WorldModel(cfg)
    wm.load()

    obs_frames, clr_frames = [], []
    print("\n[Anchor Builder]  'o' = obstacle  'c' = clear  'q' = quit")

    while True:
        frame = cam_buf.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr,
                    f"obs={len(obs_frames)} clear={len(clr_frames)} | o/c/q",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Anchor Builder", bgr)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("o"):
            obs_frames.append(frame)
        elif key == ord("c"):
            clr_frames.append(frame)
        elif key == ord("q"):
            break

    cam_buf.stop()
    cv2.destroyAllWindows()

    if obs_frames and clr_frames:
        wm.build_anchors(obs_frames, clr_frames)
        print("Anchors updated.")
    else:
        print("Not enough frames – anchors unchanged.")


if __name__ == "__main__":
    main()
