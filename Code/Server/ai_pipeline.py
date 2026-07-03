"""
ai_pipeline.py – AI pipeline orchestrator (server side).

This class owns the full perception → prediction → decision → action loop and
runs it in a background thread so it does not block the Freenove TCP server.

Thread-safe state is exposed via AIState (a dataclass protected by a lock) so
the TCP layer can broadcast live status to connected clients without races.

Pipeline per frame:
  1. Get latest frame from CameraBuffer
  2. Detector.detect()      → DetectionResult  (YOLOv8, every N frames)
  3. WorldModel.predict()   → WorldModelResult (V-JEPA 2, every M frames)
  4. TemporalAction.push()  → TemporalResult   (rules, every frame)
  5. DecisionFuser.decide() → DecisionResult   (weighted fusion + hysteresis)
  6. RobotController.execute() → motor commands
  7. Visualizer.annotate()  → BGR overlay frame
  8. NavigationLogger.log_frame()
  9. Update AIState for TCP broadcast

The pipeline runs in predictive OR baseline mode.  Switching mode at runtime
is supported via set_navigation_mode() (called from the TCP command handler).
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AIState:
    """Thread-safe snapshot of the current AI pipeline output."""
    action: str = "STOP"
    risk_score: float = 0.0
    world_model_label: str = "UNKNOWN"
    temporal_pattern: str = "UNKNOWN"
    navigation_mode: str = "predictive"
    obstacles_detected: int = 0
    ultrasonic_cm: float = -1.0
    frame_count: int = 0
    running: bool = False


class AIPipeline:
    """
    Runs the full perception-prediction-decision loop in a daemon thread.

    Designed to sit alongside the existing Freenove TankServer and Camera
    objects without replacing them.
    """

    def __init__(self, cfg: dict | None = None, cfg_path: str = "config.yaml"):
        if cfg is not None:
            self._cfg = cfg
        else:
            with open(cfg_path) as f:
                self._cfg = yaml.safe_load(f)

        self._nav_mode = self._cfg.get("navigation_mode", "predictive")

        self._state = AIState(navigation_mode=self._nav_mode)
        self._state_lock = threading.Lock()
        self._motor_enabled = True  # False when UI disables AI (CMD_AIMODE#0)

        self._last_annotated_bgr: np.ndarray | None = None
        self._annotated_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._running = False

        # Components (initialised in start())
        self._cam_buf = None
        self._detector = None
        self._world_model = None
        self._temporal = None
        self._fuser = None
        self._robot = None
        self._nav_logger = None
        self._visualizer = None

        # Freenove objects (live-Pi mode)
        self._freenove_camera = None
        self._freenove_car = None
        self._tcp_server = None

        # Split-inference mode objects
        self._robot_connection = None   # RobotConnectionServer (TCP to Pi)
        self._ext_cam_buf = None        # pre-built CameraBuffer (shared with robot_connection)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self, freenove_camera=None, freenove_car=None, tcp_server=None,
               robot_connection=None, camera_buffer=None) -> None:
        """
        Attach hardware / connection objects before calling start().

        In classic live mode pass freenove_camera, freenove_car, tcp_server.
        In split-inference mode pass robot_connection and (optionally) a
        pre-built camera_buffer shared with robot_connection.
        """
        self._freenove_camera = freenove_camera
        self._freenove_car = freenove_car
        self._tcp_server = tcp_server
        self._robot_connection = robot_connection
        self._ext_cam_buf = camera_buffer

    def start(self) -> None:
        self._build_components()
        self._running = True
        with self._state_lock:
            self._state.running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AIPipeline"
        )
        self._thread.start()
        logger.info("AI pipeline started (mode=%s)", self._nav_mode)

    def stop(self) -> None:
        self._running = False
        with self._state_lock:
            self._state.running = False
        if self._robot:
            self._robot.safe_stop()
        if self._cam_buf and self._ext_cam_buf is None:
            self._cam_buf.stop()
        if self._nav_logger:
            self._nav_logger.close()
        if self._visualizer:
            self._visualizer.close()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("AI pipeline stopped")

    # ── Runtime controls ──────────────────────────────────────────────────────

    def set_motor_enabled(self, enabled: bool) -> None:
        """Enable or disable motor output without stopping the AI pipeline."""
        with self._state_lock:
            self._motor_enabled = enabled
        logger.info("Motor output %s", "enabled" if enabled else "disabled (manual/off mode)")

    def set_navigation_mode(self, mode: str) -> None:
        """Switch between 'predictive' and 'baseline' at runtime."""
        if mode not in ("predictive", "baseline"):
            logger.warning("Unknown navigation mode: %s", mode)
            return
        self._nav_mode = mode
        # Rebuild only the fuser (cheap) to swap the weights
        from decision import DecisionFuser
        self._fuser = DecisionFuser(self._cfg, mode)
        with self._state_lock:
            self._state.navigation_mode = mode
        logger.info("Navigation mode switched to: %s", mode)

    def get_state(self) -> AIState:
        with self._state_lock:
            # Return a shallow copy so caller doesn't mutate shared state
            return AIState(**self._state.__dict__)

    def get_annotated_jpeg(self) -> bytes | None:
        """Return the latest annotated frame as JPEG bytes for TCP streaming."""
        with self._annotated_lock:
            bgr = self._last_annotated_bgr
        if bgr is None:
            return None
        return self._visualizer.encode_jpeg(bgr)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        frame_idx = 0
        no_frame_ticks = 0
        while self._running:
            # ── 1. Grab latest frame ──────────────────────────────────────────
            frame_rgb = self._cam_buf.get_latest_frame()
            if frame_rgb is None:
                no_frame_ticks += 1
                # Log once per ~5 s so the operator can see frames are missing
                if no_frame_ticks % 250 == 1:
                    logger.warning(
                        "AI pipeline waiting for camera frames – none in buffer yet. "
                        "Check the robot camera stream (port 8004)."
                    )
                time.sleep(0.02)
                continue
            no_frame_ticks = 0

            try:
                self._process_frame(frame_rgb, frame_idx + 1)
                frame_idx += 1
            except Exception as exc:
                logger.exception("AI pipeline frame error (continuing): %s", exc)
                time.sleep(0.02)

    def _process_frame(self, frame_rgb, frame_idx: int) -> None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ── 2. Detection (local YOLO in demo mode; CMD_DETECTION in live mode)
            if self._robot_connection is not None and self._robot_connection.is_connected:
                # Pi ran YOLOv8n and sent CMD_DETECTION – use that result
                det_result = self._robot_connection.get_latest_detection()
            elif self._detector is not None:
                try:
                    det_result = self._detector.detect(frame_rgb)
                except Exception as exc:
                    logger.warning("Detector error: %s", exc)
                    return
            else:
                from detector import DetectionResult
                det_result = DetectionResult()  # empty – no detection source

            # ── 3. V-JEPA 2 world model prediction ───────────────────────────
            clip = self._cam_buf.get_clip()
            if clip:
                wm_result = self._world_model.predict(clip)
            else:
                from world_model import WorldModelResult
                wm_result = WorldModelResult(buffer_ready=False,
                                             predicted_risk=det_result.raw_risk)

            wm_risk = (wm_result.predicted_risk
                       if wm_result.buffer_ready else det_result.raw_risk)

            # ── 4. Temporal motion pattern ────────────────────────────────────
            obs_state = self._detection_to_state(det_result)
            self._temporal.push(obs_state)
            temporal_result = self._temporal.classify()

            # ── 5. Ultrasonic guard (hard safety) ─────────────────────────────
            sonic_risk = self._robot.get_ultrasonic_risk()
            sonic_cm = -1.0
            if self._robot_connection:
                sonic_cm = self._robot_connection.get_sonic_cm()
            elif self._freenove_car:
                try:
                    sonic_cm = self._freenove_car.sonic.get_distance()
                except Exception:
                    pass

            # ── 6. Decision fusion ────────────────────────────────────────────
            decision = self._fuser.decide(
                detector_risk=det_result.raw_risk,
                world_model_risk=wm_risk,
                temporal_risk=temporal_result.temporal_risk,
                world_model_label=wm_result.label,
                temporal_pattern=temporal_result.pattern,
                ultrasonic_risk=sonic_risk,
            )

            logger.info(
                "[%05d] %s risk=%.2f | det=%.2f wm=%.2f ta=%.2f | %s",
                frame_idx, decision.action, decision.risk_score,
                det_result.raw_risk, wm_risk, temporal_result.temporal_risk,
                decision.explanation,
            )

            # ── 7. Execute motor command ──────────────────────────────────────
            with self._state_lock:
                motor_enabled = self._motor_enabled
            if motor_enabled:
                self._execute_action(self._robot, decision.action)

            # ── 8. Visualise ──────────────────────────────────────────────────
            annotated = self._visualizer.annotate(
                frame_bgr, det_result, decision, temporal_result, sonic_cm
            )
            with self._annotated_lock:
                self._last_annotated_bgr = annotated
            self._visualizer.show(annotated)

            # ── 9. Log ────────────────────────────────────────────────────────
            self._nav_logger.log_frame(
                annotated, decision, det_result, sonic_cm
            )

            # ── 10. Broadcast AI status to TCP client ─────────────────────────
            self._broadcast_status(decision, temporal_result, sonic_cm)

            # ── 11. Update shared state ───────────────────────────────────────
            with self._state_lock:
                self._state.action = decision.action
                self._state.risk_score = decision.risk_score
                self._state.world_model_label = wm_result.label
                self._state.temporal_pattern = temporal_result.pattern
                self._state.obstacles_detected = len(det_result.boxes)
                self._state.ultrasonic_cm = sonic_cm
                self._state.frame_count = frame_idx

    # ── Component construction ────────────────────────────────────────────────

    def _build_components(self) -> None:
        from ai_logger import NavigationLogger
        from camera_buffer import CameraBuffer
        from decision import DecisionFuser
        from detector import Detector
        from robot_control import build_robot_controller
        from temporal_action import TemporalActionRecognizer
        from visualization import Visualizer
        from world_model import WorldModel

        cfg = self._cfg

        # Use pre-built CameraBuffer (split-inference / tcp mode) or create one
        if self._ext_cam_buf is not None:
            self._cam_buf = self._ext_cam_buf
        else:
            self._cam_buf = CameraBuffer(cfg, freenove_camera=self._freenove_camera)
            self._cam_buf.start()

        # Only load local YOLO in demo mode; live mode uses CMD_DETECTION from Pi
        if self._robot_connection is None:
            self._detector = Detector(cfg)
            self._detector.load()
        else:
            self._detector = None
            logger.info("Local YOLOv8 skipped – using CMD_DETECTION from Pi")

        self._world_model = WorldModel(cfg)
        self._world_model.load()

        self._temporal = TemporalActionRecognizer(cfg)
        self._fuser = DecisionFuser(cfg, self._nav_mode)
        # Use TCP controller when robot_connection available, else real/mock
        self._robot = build_robot_controller(cfg, self._freenove_car, self._robot_connection)
        from temporal_action import detection_to_state
        from robot_control import execute_action
        self._detection_to_state = detection_to_state
        self._execute_action = execute_action
        self._nav_logger = NavigationLogger(cfg, self._nav_mode)
        self._visualizer = Visualizer(cfg, self._nav_mode)

    # ── TCP status broadcast ──────────────────────────────────────────────────

    def _broadcast_status(self, decision, temporal_result, sonic_cm: float) -> None:
        """
        Send a compact status string to all connected command clients so the
        lightweight client UI can display live AI state without video decoding.

        Format: CMD_AISTATUS#<action>#<risk*100>#<wm_label>#<pattern>#<sonic_cm>\r\n
        """
        if self._tcp_server is None:
            return
        try:
            msg = (
                f"CMD_AISTATUS#{decision.action}"
                f"#{int(decision.risk_score * 100)}"
                f"#{decision.world_model_label}"
                f"#{temporal_result.pattern}"
                f"#{sonic_cm:.1f}\r\n"
            )
            if self._tcp_server.isCmdServerConnected():
                self._tcp_server.sendDataToCmdClinet(msg)
        except Exception as exc:
            logger.debug("Status broadcast error: %s", exc)
