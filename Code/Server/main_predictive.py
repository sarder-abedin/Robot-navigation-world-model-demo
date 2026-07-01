"""
main_predictive.py – New server entry point for the predictive navigation demo.

This module EXTENDS the existing Freenove server architecture rather than
replacing it.  It reuses:
  TankServer  (TCP cmd + video server from server.py / tcp_server.py)
  Car         (motor, servo, ultrasonic, infrared from car.py)
  Camera      (picamera2 streaming from camera.py)
  Led         (LED strip from led.py)
  Command     (command constants from command.py)
  MessageParser (from message.py)

New additions (alongside, not replacing):
  AIPipeline  – runs the full AI loop in a background thread
  CMD_AIMODE  – new command to switch predictive/baseline/off from client

Usage:
  # Live mode (on the Raspberry Pi):
  python main_predictive.py

  # Demo mode (laptop, no hardware):
  python main_predictive.py --mode demo --nav predictive
  python main_predictive.py --mode demo --nav baseline

  # Build calibration anchors for V-JEPA 2:
  python main_predictive.py --build-anchors

Press 'q' in the visualisation window (if attached) to quit gracefully.
Ctrl-C always performs a safe stop.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import struct
import sys
import threading
import time

import yaml

logger = logging.getLogger(__name__)

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown = False


def _handle_signal(sig, frame):
    global _shutdown
    logger.warning("Signal %s received – shutting down…", sig)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Predictive Navigation Server")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mode", choices=["demo", "live"], default=None,
                   help="Override config: 'demo' (video file) or 'live' (Pi camera)")
    p.add_argument("--nav", choices=["predictive", "baseline"], default=None,
                   help="Navigation intelligence mode")
    p.add_argument("--video", default=None, help="Demo video path override")
    p.add_argument("--build-anchors", action="store_true",
                   help="Calibrate V-JEPA 2 anchors interactively then exit")
    p.add_argument("--no-display", action="store_true",
                   help="Disable OpenCV window even if a display is available")
    return p.parse_args()


# ── Config loader ──────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Live Freenove server wrapper ───────────────────────────────────────────────

class PredictiveServer:
    """
    Wraps the Freenove TankServer with an AI pipeline thread.

    The existing Freenove threading structure is preserved:
      threading_cmd_receive   – handles CMD_MOTOR, CMD_SERVO, CMD_LED, etc.
      threading_video_send    – sends annotated JPEG frames to client
      AIPipeline thread       – handles AI inference and motor commands

    When the AI pipeline is active, CMD_MOTOR from the client is silently
    ignored (the AI controls the motors).  CMD_AIMODE switches control back.
    """

    # New commands added by this project
    CMD_AIMODE   = "CMD_AIMODE"    # #0 stop AI  #1 baseline  #2 predictive
    CMD_AISTATUS = "CMD_AISTATUS"  # sent TO client: action#risk#wm#pattern#sonic
    CMD_KILL     = "CMD_KILL"      # #0 full server shutdown
    CMD_AIMOVE   = "CMD_AIMOVE"    # FORWARD/SLOW/STOP/REROUTE — from client PC AI

    def __init__(self, cfg: dict, nav_mode: str):
        self._cfg = cfg
        self._nav_mode = nav_mode
        self._ai_active = True

        # Freenove infrastructure
        from server import TankServer
        from command import Command
        from message import MessageParser

        self._tcp = TankServer()
        self._command = Command()
        self._parser = MessageParser()

        # Freenove hardware (only in live mode)
        self._car = None
        self._camera = None
        self._led = None

        # AI pipeline
        from ai_pipeline import AIPipeline
        self._ai = AIPipeline(cfg_path=args.config if "args" in dir() else "config.yaml")

        # Threading flags
        self._cmd_running = False
        self._video_running = False
        self._cmd_thread = None
        self._video_thread = None

    def start(self) -> None:
        mode = self._cfg.get("mode", "demo")

        # ── Hardware (live mode only) ─────────────────────────────────────────
        if mode == "live":
            from car import Car
            from camera import Camera
            from led import Led
            self._car = Car()
            self._camera = Camera(stream_size=(
                self._cfg["camera"]["stream_width"],
                self._cfg["camera"]["stream_height"],
            ))
            self._led = Led()
            self._camera.start_stream()

        # ── Attach hardware to AI pipeline ────────────────────────────────────
        srv_cfg = self._cfg.get("server", {})
        self._tcp.startTcpServer(
            port1=srv_cfg.get("cmd_port", 5003),
            port2=srv_cfg.get("video_port", 8003),
        )
        self._ai.attach(
            freenove_camera=self._camera,
            freenove_car=self._car,
            tcp_server=self._tcp,
        )
        self._ai.start()

        # ── TCP threads ───────────────────────────────────────────────────────
        self._cmd_running = True
        self._cmd_thread = threading.Thread(
            target=self._cmd_loop, daemon=True, name="CmdReceive"
        )
        self._cmd_thread.start()

        self._video_running = True
        self._video_thread = threading.Thread(
            target=self._video_loop, daemon=True, name="VideoSend"
        )
        self._video_thread.start()

        logger.info(
            "PredictiveServer started – cmd:%d  video:%d  nav:%s",
            srv_cfg.get("cmd_port", 5003),
            srv_cfg.get("video_port", 8003),
            self._nav_mode,
        )

    def stop(self) -> None:
        self._cmd_running = False
        self._video_running = False
        self._ai.stop()
        self._tcp.stopTcpServer()
        if self._camera:
            self._camera.stop_stream()
            self._camera.close()
        if self._car:
            self._car.close()
        if self._led:
            self._led.colorWipe([0, 0, 0])
        logger.info("PredictiveServer stopped")

    # ── Command receive thread ─────────────────────────────────────────────────

    def _cmd_loop(self) -> None:
        while self._cmd_running:
            queue = self._tcp.readDataFromCmdServer()
            while queue.qsize() > 0:
                _, raw = queue.get()
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
            # 0 = stop AI  1 = baseline  2 = predictive
            val = self._parser.intParameter[0] if self._parser.intParameter else -1
            if val == 0:
                self._ai_active = False
                if self._car:
                    self._car.motor.setMotorModel(0, 0)
                logger.info("AI control disabled by client")
            elif val == 1:
                self._ai_active = True
                self._ai.set_navigation_mode("baseline")
            elif val == 2:
                self._ai_active = True
                self._ai.set_navigation_mode("predictive")

        elif cmd == self._command.CMD_MOTOR:
            # Only honour manual motor commands when AI is off
            if not self._ai_active and self._car and len(self._parser.intParameter) >= 2:
                l, r = self._parser.intParameter[0], self._parser.intParameter[1]
                self._car.motor.setMotorModel(l, r)

        elif cmd == self._command.CMD_SERVO:
            if self._car and len(self._parser.intParameter) >= 2:
                idx = self._parser.intParameter[0]
                angle = self._parser.intParameter[1]
                self._car.servo.setServoAngle(idx, angle)

        elif cmd == self.CMD_AIMOVE:
            # Navigation command from client PC AI (V-JEPA2+SSv2+decision on client)
            parts = msg.split("#")
            action_str = parts[1].strip() if len(parts) > 1 else "STOP"
            if self._ai_active:
                self._ai.apply_client_action(action_str)

        elif cmd == self.CMD_KILL:
            # Client emergency kill: stop motors and shut down the server process
            logger.warning("CMD_KILL received from client – initiating shutdown")
            if self._car:
                self._car.motor.setMotorModel(0, 0)
            self._ai.stop()
            global _shutdown
            _shutdown = True

        elif cmd == self._command.CMD_MODE:
            # Mode 0 = manual (disables AI), others re-enable
            val = self._parser.intParameter[0] if self._parser.intParameter else -1
            if val == 0:
                self._ai_active = False
                if self._car:
                    self._car.motor.setMotorModel(0, 0)

    # ── Video send thread ──────────────────────────────────────────────────────

    def _video_loop(self) -> None:
        """
        Send annotated JPEG frames to the connected video client.

        When the AI pipeline has produced an annotated frame we encode it as
        JPEG and send it instead of the raw camera JPEG, giving the client a
        live view of detections, risk, and action overlays.
        """
        while self._video_running:
            if not self._tcp.isVideoServerConnected():
                time.sleep(0.1)
                continue
            try:
                jpg = self._ai.get_annotated_jpeg()
                if jpg is None:
                    # Fall back to raw camera frame while AI is warming up
                    if self._camera and self._camera.streaming:
                        jpg = self._camera.get_frame()
                if jpg:
                    length_bin = struct.pack("<I", len(jpg))
                    self._tcp.sendDataToVideoClient(length_bin)
                    self._tcp.sendDataToVideoClient(jpg)
            except Exception as exc:
                logger.debug("Video send error: %s", exc)
            time.sleep(1.0 / 20)  # target 20 fps to the client


# ── Anchor builder utility ─────────────────────────────────────────────────────

def run_anchor_builder(cfg: dict) -> None:
    """
    Interactive tool to collect labelled frames and update V-JEPA 2 anchors.
    Press 'o' → obstacle, 'c' → clear, 'q' → finish.
    """
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

        import cv2
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr,
                    f"obs={len(obs_frames)} clear={len(clr_frames)} | o/c/q",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Anchor Builder", bgr)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("o"):
            obs_frames.append(frame)
            print(f"  obstacle frame captured ({len(obs_frames)})")
        elif key == ord("c"):
            clr_frames.append(frame)
            print(f"  clear frame captured ({len(clr_frames)})")
        elif key == ord("q"):
            break

    cam_buf.stop()
    cv2.destroyAllWindows()

    if obs_frames and clr_frames:
        wm.build_anchors(obs_frames, clr_frames)
        print("Anchors updated.")
    else:
        print("Not enough frames – anchors unchanged.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global args
    args = parse_args()
    cfg = load_cfg(args.config)

    # CLI overrides
    if args.mode:
        cfg["mode"] = args.mode
    if args.video:
        cfg["camera"]["demo_video_path"] = args.video
    if args.no_display:
        cfg["visualization"]["show_window"] = False

    nav_mode = args.nav or cfg.get("navigation_mode", "predictive")

    # Set up root logging early
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.build_anchors:
        run_anchor_builder(cfg)
        return

    server = PredictiveServer(cfg, nav_mode)
    server.start()

    print(f"\n{'='*60}")
    print(f"  Freenove Predictive Navigation Server")
    print(f"  Mode: {cfg.get('mode','demo')}  |  Navigation: {nav_mode}")
    print(f"  CMD port: {cfg.get('server',{}).get('cmd_port',5003)}")
    print(f"  Video port: {cfg.get('server',{}).get('video_port',8003)}")
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
        logger.info("Server shut down cleanly")


if __name__ == "__main__":
    main()
