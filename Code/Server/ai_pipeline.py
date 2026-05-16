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

    def __init__(self, cfg_path: str = "config.yaml"):
        with open(cfg_path) as f:
            self._cfg = yaml.safe_load(f)

        self._mode = cfg_path  # keep path for reloads
        self._nav_mode = self._cfg.get("navigation_mode", "predictive")

        self._state = AIState(navigation_mode=self._nav_mode)
        self._state_lock = threading.Lock()

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

        # Freenove objects passed in from the server
        self._freenove_camera = None
        self._freenove_car = None
        self._tcp_server = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self, freenove_camera=None, freenove_car=None, tcp_server=None) -> None:
        """Attach the Freenove hardware objects before calling start()."""
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
        logger.info("AI pipeline started (mode=%s)", self._nav_mode)

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
        if self._last_annotated_bgr is None:
            return None
        return self._visualizer.encode_jpeg(self._last_annotated_bgr)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        frame_idx = 0
        while self._running:
            # ── 1. Grab latest frame ──────────────────────────────────────────
            frame_rgb = self._cam_buf.get_latest_frame()
            if frame_rgb is None:
                time.sleep(0.02)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_idx += 1

            # ── 2. YOLOv8 detection ───────────────────────────────────────────
            try:
                det_result = self._detector.detect(frame_rgb)
            except Exception as exc:
                logger.warning("Detector error: %s", exc)
                continue

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
            from temporal_action import detection_to_state
            obs_state = detection_to_state(det_result)
            self._temporal.push(obs_state)
            temporal_result = self._temporal.classify()

            # ── 5. Ultrasonic guard (hard safety) ─────────────────────────────
            sonic_risk = self._robot.get_ultrasonic_risk()
            sonic_cm = -1.0
            if self._freenove_car:
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
            from robot_control import execute_action
            execute_action(self._robot, decision.action)

            # ── 8. Visualise ──────────────────────────────────────────────────
            annotated = self._visualizer.annotate(
                frame_bgr, det_result, decision, temporal_result, sonic_cm
            )
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

        self._cam_buf = CameraBuffer(cfg, freenove_camera=self._freenove_camera)
        self._cam_buf.start()

        self._detector = Detector(cfg)
        self._detector.load()

        self._world_model = WorldModel(cfg)
        self._world_model.load()

        self._temporal = TemporalActionRecognizer(cfg)
        self._fuser = DecisionFuser(cfg, self._nav_mode)
        self._robot = build_robot_controller(cfg, self._freenove_car)
        self._nav_logger = NavigationLogger(cfg, self._nav_mode)
        self._visualizer = Visualizer(cfg, self._nav_mode)
        self._last_annotated_bgr: np.ndarray | None = None

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
