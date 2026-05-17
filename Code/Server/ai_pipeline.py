"""
ai_pipeline.py – Detection pipeline orchestrator (server / Raspberry Pi side).

Architecture (post-refactor)
────────────────────────────
The Pi is responsible for:
  1. CameraBuffer    – rolling frame capture (live picamera2 or demo video)
  2. Detector        – YOLOv8 obstacle detection (every N frames)
  3. RobotController – ultrasonic safety guard + motor execution
  4. CMD_DETECTION   – broadcast YOLOv8 results to the client PC each frame
  5. apply_client_action() – execute CMD_AIMOVE received from the client PC

V-JEPA 2 world model, SSv2 temporal reasoning, and decision fusion now run
on the client PC (Code/Client/ai_viewer.py) where GPU headroom is available.
The client closes the loop by sending CMD_AIMOVE#<ACTION> back to the Pi.

TCP messages (new additions):
  Pi → Client:  CMD_DETECTION#<yolo_risk_pct>#<obs_in_center>#<area_frac_pct>
                              #<centroid_x_pct>#<sonic_cm>\r\n
  Client → Pi:  CMD_AIMOVE#FORWARD | SLOW | STOP | REROUTE  (handled in main_predictive.py)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AIState:
    """Thread-safe snapshot of the Pi-side pipeline state."""
    last_client_action: str = "STOP"
    yolo_risk: float = 0.0
    obstacles_detected: int = 0
    ultrasonic_cm: float = -1.0
    frame_count: int = 0
    navigation_mode: str = "predictive"
    running: bool = False


class AIPipeline:
    """
    Runs the Pi-side detection loop in a background daemon thread.

    The server no longer makes navigation decisions — it only detects obstacles
    and forwards detection data to the client, then executes whatever action
    the client sends back via CMD_AIMOVE.
    """

    def __init__(self, cfg_path: str = "config.yaml"):
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)
        self._cfg_path = cfg_path
        self._nav_mode = self._cfg.get("navigation_mode", "predictive")

        self._state = AIState(navigation_mode=self._nav_mode)
        self._state_lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._last_client_action = "STOP"

        self._thread: threading.Thread | None = None
        self._running = False
        self._last_annotated_bgr: np.ndarray | None = None

        # Components (built in start())
        self._cam_buf = None
        self._detector = None
        self._robot = None
        self._nav_logger = None
        self._visualizer = None

        # Freenove objects (attached before start())
        self._freenove_camera = None
        self._freenove_car = None
        self._tcp_server = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self, freenove_camera=None, freenove_car=None, tcp_server=None) -> None:
        self._freenove_camera = freenove_camera
        self._freenove_car = freenove_car
        self._tcp_server = tcp_server

    def start(self) -> None:
        self._build_components()
        self._running = True
        with self._state_lock:
            self._state.running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AIPipeline"
        )
        self._thread.start()
        logger.info(
            "AI pipeline started – detection on Pi, world model + decision on client"
        )

    def stop(self) -> None:
        self._running = False
        with self._state_lock:
            self._state.running = False
        if self._robot:
            self._robot.safe_stop()
        if self._cam_buf:
            self._cam_buf.stop()
        if self._nav_logger:
            self._nav_logger.close()
        if self._visualizer:
            self._visualizer.close()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("AI pipeline stopped")

    # ── Runtime controls ──────────────────────────────────────────────────────

    def set_navigation_mode(self, mode: str) -> None:
        """Update mode label (actual weights live on the client DecisionFuser)."""
        self._nav_mode = mode
        with self._state_lock:
            self._state.navigation_mode = mode
        logger.info("Navigation mode updated to: %s", mode)

    def apply_client_action(self, action_str: str) -> None:
        """
        Execute a navigation action sent by the client PC via CMD_AIMOVE.

        The client PC runs V-JEPA 2 + SSv2 + decision fusion and sends the
        resulting action string here.  The Pi converts it to motor commands.
        """
        from robot_control import execute_action, Action
        with self._action_lock:
            self._last_client_action = action_str
        try:
            action = Action(action_str)
            execute_action(self._robot, action)
        except Exception as exc:
            logger.debug("apply_client_action error (%s): %s", action_str, exc)

    def get_state(self) -> AIState:
        with self._state_lock:
            return AIState(**self._state.__dict__)

    def get_annotated_jpeg(self) -> bytes | None:
        if self._last_annotated_bgr is None:
            return None
        return self._visualizer.encode_jpeg(self._last_annotated_bgr)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        frame_idx = 0
        while self._running:
            # 1. Grab latest frame
            frame_rgb = self._cam_buf.get_latest_frame()
            if frame_rgb is None:
                time.sleep(0.02)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_idx += 1

            # 2. YOLOv8 detection
            try:
                det_result = self._detector.detect(frame_rgb)
            except Exception as exc:
                logger.warning("Detector error: %s", exc)
                continue

            # 3. Ultrasonic safety guard (hard stop on the Pi regardless of client)
            sonic_risk = self._robot.get_ultrasonic_risk()
            sonic_cm = -1.0
            if self._freenove_car:
                try:
                    sonic_cm = self._freenove_car.sonic.get_distance()
                except Exception:
                    pass

            if sonic_risk >= 1.0:
                self._robot.stop()
                with self._action_lock:
                    self._last_client_action = "STOP"
                logger.debug("Ultrasonic safety stop (%.1f cm)", sonic_cm)

            # 4. Broadcast detection to client for world model + decision
            self._broadcast_detection(det_result, sonic_cm)

            # 5. Annotate frame (YOLO boxes + last client action + sonic)
            with self._action_lock:
                client_action = self._last_client_action
            annotated = self._visualizer.annotate(
                frame_bgr, det_result, sonic_cm, client_action
            )
            self._last_annotated_bgr = annotated
            self._visualizer.show(annotated)

            # 6. Log
            self._nav_logger.log_detection_frame(
                annotated, det_result, sonic_cm, client_action
            )

            # 7. Update shared state
            with self._state_lock:
                self._state.last_client_action = client_action
                self._state.yolo_risk = det_result.raw_risk
                self._state.obstacles_detected = len(det_result.boxes)
                self._state.ultrasonic_cm = sonic_cm
                self._state.frame_count = frame_idx

    # ── Component construction ────────────────────────────────────────────────

    def _build_components(self) -> None:
        from ai_logger import NavigationLogger
        from camera_buffer import CameraBuffer
        from detector import Detector
        from robot_control import build_robot_controller
        from visualization import Visualizer

        cfg = self._cfg
        self._cam_buf = CameraBuffer(cfg, freenove_camera=self._freenove_camera)
        self._cam_buf.start()

        self._detector = Detector(cfg)
        self._detector.load()

        self._robot = build_robot_controller(cfg, self._freenove_car)
        self._nav_logger = NavigationLogger(cfg, self._nav_mode)
        self._visualizer = Visualizer(cfg, self._nav_mode)

    # ── TCP broadcast ─────────────────────────────────────────────────────────

    def _broadcast_detection(self, det_result, sonic_cm: float) -> None:
        """
        Broadcast per-frame YOLOv8 results to the client PC.

        Format (all fields delimited by #):
          CMD_DETECTION
            #<yolo_risk_pct>      int 0-100  (det_result.raw_risk * 100)
            #<obs_in_center>      0 or 1
            #<area_frac_pct>      int 0-100  (closest bbox area fraction * 100)
            #<centroid_x_pct>     int 0-100  (mean horizontal centroid, 50 = centre)
            #<sonic_cm>           float or -1.0

        The client uses these fields to:
          - Feed yolo_risk into the DecisionFuser as detector_risk
          - Reconstruct a FrameObstacleState for the TemporalActionRecognizer
          - Display sonic distance in the UI
        """
        if self._tcp_server is None:
            return
        try:
            if det_result.boxes:
                w = det_result.frame_width or 400
                cxs = [
                    (x1 + x2) / 2 / w
                    for x1, _y1, x2, _y2 in det_result.boxes
                ]
                centroid_x_pct = int(float(np.mean(cxs)) * 100)
            else:
                centroid_x_pct = 50

            msg = (
                f"CMD_DETECTION"
                f"#{int(det_result.raw_risk * 100)}"
                f"#{int(det_result.obstacle_in_center)}"
                f"#{int(det_result.closest_area * 100)}"
                f"#{centroid_x_pct}"
                f"#{sonic_cm:.1f}\r\n"
            )
            if self._tcp_server.isCmdServerConnected():
                self._tcp_server.sendDataToCmdClinet(msg)
        except Exception as exc:
            logger.debug("Detection broadcast error: %s", exc)
