"""
ai_pipeline.py – AI pipeline orchestrator (server side).

This class owns the full perception → prediction → decision → action loop and
runs it in a background thread so it does not block the Freenove TCP server.

Thread-safe state is exposed via AIState (a dataclass protected by a lock) so
the TCP layer can broadcast live status to connected clients without races.

Pipeline per frame:
  1. Get latest frame from CameraBuffer
  2. Detector.detect()      → DetectionResult  (YOLO11, every N frames)
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
    ssv2_sentence: str = ""       # genuine SSv2 label with YOLO object filled in
    ssv2_confidence: float = 0.0
    logging_enabled: bool = False


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

        # EMA of the per-frame processing time (perception + decision), used as
        # the "reaction time" the speed governor reserves braking distance for.
        # Seeded from the governor's min_reaction so the first frames aren't
        # over-optimistic on a slow (CPU) pipeline.
        gov_cfg = (self._cfg.get("decision", {}) or {}).get("governor", {}) or {}
        self._reaction_ema = float(gov_cfg.get("min_reaction_s", 0.5))
        # Cumulative count of camera frames received but skipped (stale) because the
        # pipeline was still busy — logged as a network/compute drop statistic.
        self._net_dropped_total = 0
        # Depth distance within which a (non-YOLO) obstacle counts as "present"
        # for the motion recogniser, so walls aren't invisible to it.
        self._depth_presence_range = float(
            (self._cfg.get("temporal_action", {}) or {}).get("depth_presence_range_m", 1.5))

        # Run logging (CSV + annotated frames) starts from config; toggled at
        # runtime from the UI (CMD_LOGGING) or at startup via env/CLI.
        self._logging_enabled = bool(self._cfg.get("logging", {}).get("enabled", False))

        self._state = AIState(navigation_mode=self._nav_mode,
                              logging_enabled=self._logging_enabled)
        self._state_lock = threading.Lock()
        # Start IDLE: the robot must not drive until the operator activates AI from
        # the UI (after selecting a goal). Safer, and matches the connect→goal→
        # activate flow. Headless/demo can force-on via --ai-start (set_motor_enabled).
        self._motor_enabled = False
        # Goal-point tracking (Phase 2): follow the user-selected point across
        # frames and report bearing + depth for the HUD. Does NOT drive motion yet.
        from goal_navigator import GoalTracker
        self._goal_tracker = GoalTracker(cfg)
        self._goal_state = None     # latest GoalState (tracked position, bearing, depth)
        self._goal_reached_handled = False   # fired the stop-on-arrival once
        self._goal_lost_handled = False      # fired the stop-on-lost once (following mode)
        self._goal_following = False         # Goal-Following mode (steer to goal) vs avoidance-only
        self._goal_center_tol = float(
            ((cfg.get("goal", {}) or {}).get("center_tolerance_deg", 12.0)))

        self._last_annotated_bgr: np.ndarray | None = None
        self._annotated_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._running = False

        # The heavy models (V-JEPA 2 + SSv2) run on their OWN thread — their ViT-L /
        # VideoMAE forwards take up to tens of seconds on MPS, far too slow for the
        # drive loop (a stall there starves CMD_AIMOVE and trips the robot's motor
        # watchdog). The worker publishes the latest results; the drive loop reads
        # whatever is current and never blocks. SSv2 is annotation-only, so a stale
        # caption is harmless; the drive loop feeds it the latest YOLO object label.
        self._wm_thread: threading.Thread | None = None
        self._wm_lock = threading.Lock()
        self._latest_wm = None            # last WorldModelResult (None until first run)
        self._latest_ssv2 = None          # last SSv2Result   (None until first run)
        self._latest_object_label = ""    # newest YOLO closest-object label for SSv2
        self._prepared = False            # models built + warmed up

        # Components (initialised in start())
        self._cam_buf = None
        self._detector = None
        self._world_model = None
        self._temporal = None
        self._ssv2 = None
        self._depth = None
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

    def prepare(self) -> None:
        """Build AND warm up every model before the pipeline drives.

        Call this before opening the robot/UI listeners: model loading + the first
        (cold) inference of each model take tens of seconds on MPS, and paying that
        here means the robot never connects to a server whose first real frames
        would each block for seconds. Idempotent.
        """
        if self._prepared:
            return
        self._build_components()
        self._warmup()
        self._prepared = True

    def _warmup(self) -> None:
        """Run one dummy inference per model so the costly first-call compile /
        graph build happens now (during startup) instead of on the first live
        frame. Best-effort — a warmup failure must never block startup."""
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        t0 = time.monotonic()
        try:
            if self._detector is not None:
                self._detector.detect(cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.debug("YOLO warmup skipped: %s", exc)
        # A clip long enough for either heavy model (V-JEPA 2 needs camera.clip_length
        # frames; SSv2 samples 16 from it).
        clip_len = int((self._cfg.get("camera", {}) or {}).get("clip_length", 64))
        clip = [dummy] * max(clip_len, 16)
        try:
            if self._world_model is not None:
                self._world_model.predict(clip)
        except Exception as exc:
            logger.debug("V-JEPA 2 warmup skipped: %s", exc)
        try:
            if self._ssv2 is not None:
                self._ssv2.recognize(clip, "")
        except Exception as exc:
            logger.debug("SSv2 warmup skipped: %s", exc)
        try:
            if self._depth is not None:
                self._depth.estimate(dummy)
        except Exception as exc:
            logger.debug("Depth warmup skipped: %s", exc)
        logger.info("Model warmup complete (%.1fs) – first live frame will be fast",
                    time.monotonic() - t0)

    def start(self) -> None:
        if not self._prepared:
            self.prepare()
        self._running = True
        with self._state_lock:
            self._state.running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AIPipeline"
        )
        self._thread.start()
        # Background worker — keeps the heavy V-JEPA 2 + SSv2 inference off the drive
        # loop so neither ever blocks CMD_AIMOVE.
        self._wm_thread = threading.Thread(
            target=self._wm_loop, daemon=True, name="Perception"
        )
        self._wm_thread.start()
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
        if self._wm_thread:
            # A V-JEPA 2 forward can be mid-flight; don't block shutdown on it.
            self._wm_thread.join(timeout=3.0)
        logger.info("AI pipeline stopped")

    # ── Runtime controls ──────────────────────────────────────────────────────

    def set_motor_enabled(self, enabled: bool) -> None:
        """Enable or disable motor output without stopping the AI pipeline."""
        with self._state_lock:
            self._motor_enabled = enabled
        logger.info("Motor output %s", "enabled" if enabled else "disabled (manual/off mode)")

    def set_goal(self, x_norm: float, y_norm: float) -> None:
        """Set the user-selected navigation goal at normalized image coords [0,1].

        Phase 2: the goal is TRACKED across frames (bearing + depth) and drawn on
        the HUD — it does NOT drive motion yet (that's a later phase).
        """
        gx = min(max(float(x_norm), 0.0), 1.0)
        gy = min(max(float(y_norm), 0.0), 1.0)
        self._goal_tracker.set_target(gx, gy)
        self._goal_reached_handled = False   # a new goal re-arms arrival detection
        self._goal_lost_handled = False
        logger.info("Navigation goal set at (%.3f, %.3f)", gx, gy)

    def clear_goal(self) -> None:
        """Clear the user-selected navigation goal."""
        self._goal_tracker.clear()
        self._goal_reached_handled = False
        self._goal_lost_handled = False
        with self._state_lock:
            self._goal_state = None
        logger.info("Navigation goal cleared")

    def set_goal_following(self, enabled: bool) -> None:
        """Enable Goal-Following mode (steer to the goal) vs obstacle-avoidance only.

        Safety/avoidance always overrides goal-seeking (see goal_navigator.goal_steering).
        """
        with self._state_lock:
            self._goal_following = bool(enabled)
        logger.info("Navigation mode: %s",
                    "GOAL FOLLOWING (steer to goal)" if enabled else "OBSTACLE AVOIDANCE")

    def get_goal_state(self):
        with self._state_lock:
            return self._goal_state

    def set_logging_enabled(self, enabled: bool) -> None:
        """Turn run logging (CSV + annotated frames) on/off at runtime."""
        with self._state_lock:
            self._logging_enabled = enabled
            self._state.logging_enabled = enabled
        logger.info("Run logging %s", "ENABLED" if enabled else "disabled")

    def is_logging_enabled(self) -> bool:
        with self._state_lock:
            return self._logging_enabled

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

    def _wm_loop(self) -> None:
        """Refresh the heavy V-JEPA 2 + SSv2 results off the drive loop.

        Both forwards (ViT-L over a 64-frame clip; VideoMAE over 16) take up to tens
        of seconds on MPS/CPU. Running them here, on one worker thread (serially, so
        we never open a second concurrent accelerator stream), means the drive loop
        never waits: it reads the latest published results and always keeps sending
        CMD_AIMOVE. A slightly-stale predictive risk / caption is fine; a frozen
        robot (watchdog STOP because CMD_AIMOVE stalled behind a blocking forward) is
        not. Note: a native MPS segfault still takes the whole process down —
        decoupling removes the *blocking*, not a hard crash.
        """
        while self._running:
            clip = self._cam_buf.get_clip() if self._cam_buf else None
            if not clip:
                time.sleep(0.05)
                continue
            if self._world_model is not None:
                try:
                    result = self._world_model.predict(clip)
                    if getattr(result, "buffer_ready", False):
                        with self._wm_lock:
                            self._latest_wm = result
                except Exception as exc:
                    logger.exception("V-JEPA 2 worker error (continuing): %s", exc)
            if self._ssv2 is not None:
                try:
                    ssv2_res = self._ssv2.recognize(clip, self._latest_object_label)
                    if getattr(ssv2_res, "buffer_ready", False):
                        with self._wm_lock:
                            self._latest_ssv2 = ssv2_res
                except Exception as exc:
                    logger.exception("SSv2 worker error (continuing): %s", exc)
            # Yield so back-to-back forwards don't peg the accelerator; predict() /
            # recognize() self-throttle via run_every_n_frames, this avoids a spin.
            time.sleep(0.02)

    def _run_loop(self) -> None:
        frame_idx = 0
        last_seq = -1
        no_frame_ticks = 0
        last_new_frame_t = time.monotonic()   # when the seq last advanced
        stale_warned = False
        while self._running:
            # ── 1. Grab latest frame (only if it's a NEW one) ─────────────────
            latest = self._cam_buf.get_latest()
            if latest is None:
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

            seq, frame_rgb = latest
            if seq == last_seq:
                # No new frame yet — don't reprocess the same one at thousands of
                # fps (that floods CMD_AIMOVE/CMD_AISTATUS and starves the UI).
                # A frame in the buffer that never advances is a *stalled* stream
                # (video channel up, frames stopped): the loop would otherwise spin
                # here silently, sending no CMD_AIMOVE, so the robot's watchdog
                # keeps STOPping it. Warn so this is visible (distinct from the
                # empty-buffer case above).
                if not stale_warned and time.monotonic() - last_new_frame_t > 3.0:
                    logger.warning(
                        "AI pipeline: camera frame is stale – no NEW frame for %.1fs "
                        "(stream connected but not updating). Robot will watchdog-STOP; "
                        "check the robot camera / video link (port 8004).",
                        time.monotonic() - last_new_frame_t,
                    )
                    stale_warned = True
                time.sleep(0.005)
                continue
            if stale_warned:
                logger.info("AI pipeline: camera frames resumed.")
                stale_warned = False
            last_new_frame_t = time.monotonic()
            # Frames that arrived while we were busy on the previous one are skipped
            # (we only ever process the latest) — count them as network/compute drops.
            if last_seq >= 0 and seq > last_seq + 1:
                self._net_dropped_total += seq - last_seq - 1
            last_seq = seq

            try:
                self._process_frame(frame_rgb, frame_idx + 1)
                frame_idx += 1
            except Exception as exc:
                logger.exception("AI pipeline frame error (continuing): %s", exc)
                time.sleep(0.02)

    def _process_frame(self, frame_rgb, frame_idx: int) -> None:
            proc_t0 = time.monotonic()
            lat: dict[str, float] = {}   # per-stage inference latency (ms)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ── 2. Detection – always the PC's local YOLO on the streamed frame.
            # Pass BGR: ultralytics treats a numpy array as BGR (OpenCV order) and
            # flips it to RGB internally, so handing it RGB swaps the R/B channels
            # and degrades detection. The Pi is a thin client (detection runs here).
            _t = time.monotonic()
            if self._detector is not None:
                try:
                    det_result = self._detector.detect(frame_bgr)
                except Exception as exc:
                    logger.warning("Detector error: %s", exc)
                    return
            else:
                from detector import DetectionResult
                det_result = DetectionResult()  # YOLO unavailable – empty result
            lat["yolo"] = (time.monotonic() - _t) * 1000.0

            # ── 3. V-JEPA 2 world model prediction ───────────────────────────
            # Read the latest result the background worker (_wm_loop) has published
            # — never run the heavy forward inline here, or it would stall the drive
            # loop and starve CMD_AIMOVE (robot watchdog-STOPs). Until the first
            # result lands, fall back to the detector's instantaneous risk.
            _t = time.monotonic()
            with self._wm_lock:
                wm_result = self._latest_wm
                ssv2_result = self._latest_ssv2
            if wm_result is None:
                from world_model import WorldModelResult
                wm_result = WorldModelResult(buffer_ready=False,
                                             predicted_risk=det_result.raw_risk)
            lat["wm"] = (time.monotonic() - _t) * 1000.0

            wm_risk = (wm_result.predicted_risk
                       if wm_result.buffer_ready else det_result.raw_risk)

            # ── 4. Depth free-space (class-agnostic geometry) ─────────────────
            # Metric distance ahead + which side is open. Sees walls the sonar and
            # YOLO miss; feeds the governor, the motion recogniser and the reroute
            # direction. Ignored (buffer_ready False) when the model isn't loaded.
            _t = time.monotonic()
            depth_result = self._depth.estimate(frame_rgb)
            depth_m = (depth_result.clear_distance_m
                       if depth_result.buffer_ready and depth_result.clear_distance_m > 0 else None)
            clear_direction = depth_result.clear_direction if depth_result.buffer_ready else None
            regions = depth_result.region_distances_m if depth_result.buffer_ready else {}
            lat["depth"] = (time.monotonic() - _t) * 1000.0

            # ── 4b. Temporal motion pattern ───────────────────────────────────
            # Use YOLO's obstacle state; but if YOLO sees nothing while depth shows
            # a close obstacle CENTRED in our path (nearer than the sides), synthesize
            # a state from depth so the motion isn't blind (stuck at STATIC_CLEAR) to
            # non-YOLO obstacles. A uniformly-close reading (open corridor with
            # mis-scaled depth, or a flat wall) is left to STATIC_CLEAR so it doesn't
            # peg temporal_risk and stall the robot on a clear path.
            obs_state = self._detection_to_state(det_result)
            if not det_result.boxes and depth_result.buffer_ready:
                from temporal_action import depth_to_obstacle_state
                ds = depth_to_obstacle_state(
                    regions.get("CENTER", depth_m), self._depth_presence_range,
                    depth_left_m=regions.get("LEFT"), depth_right_m=regions.get("RIGHT"),
                )
                if ds is not None:
                    obs_state = ds
            _t = time.monotonic()
            self._temporal.push(obs_state)
            temporal_result = self._temporal.classify()
            lat["temporal"] = (time.monotonic() - _t) * 1000.0

            # ── 4b. Genuine SSv2 action recognition (annotation/log only) ─────
            # Runs on the background worker (it's too slow for the drive loop); here
            # we just publish the newest object label for it and read its latest
            # caption. The "something" slot is filled with the CLOSEST/largest
            # obstacle's YOLO class. Does NOT affect navigation.
            object_label = getattr(det_result, "closest_label", "") or (
                det_result.labels[0] if det_result.labels else ""
            )
            self._latest_object_label = object_label
            lat["ssv2"] = 0.0    # inference is off-loop now (see _wm_loop)
            if ssv2_result is not None and getattr(ssv2_result, "buffer_ready", False):
                ssv2_sentence = ssv2_result.sentence
                ssv2_conf = ssv2_result.confidence
            else:
                ssv2_sentence = ""
                ssv2_conf = 0.0

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
            # clear_distance_m: the most conservative metric distance ahead for
            # the governor — the nearer of the ultrasonic and the depth free-space
            # (either may be None: sonar blind, or depth model absent).
            sonic_m = sonic_cm / 100.0 if sonic_cm and sonic_cm > 0 else None
            candidates = [d for d in (sonic_m, depth_m) if d is not None]
            clear_distance_m = min(candidates) if candidates else None
            # Per-side depth free-space (computed above) + the closest obstacle's
            # YOLO label feed the closed-loop reroute (wait / turn / back up).
            _t = time.monotonic()
            decision = self._fuser.decide(
                detector_risk=det_result.raw_risk,
                world_model_risk=wm_risk,
                temporal_risk=temporal_result.temporal_risk,
                world_model_label=wm_result.label,
                temporal_pattern=temporal_result.pattern,
                ultrasonic_risk=sonic_risk,
                clear_distance_m=clear_distance_m,
                reaction_s=self._reaction_ema,
                clear_direction=clear_direction,
                obstacle_label=getattr(det_result, "closest_label", "") or "",
                depth_left_m=regions.get("LEFT"),
                depth_center_m=regions.get("CENTER"),
                depth_right_m=regions.get("RIGHT"),
            )
            lat["decision"] = (time.monotonic() - _t) * 1000.0

            # Scene understanding: metric geometry (depth, class-agnostic → sees
            # walls) + the object's YOLO label when it's a known class.
            if depth_result.buffer_ready:
                obj = det_result.closest_label or "obstacle/wall"
                logger.info(
                    "[%05d] SCENE: %s %.2fm ahead | open=%s | regions=%s",
                    frame_idx, obj, depth_result.clear_distance_m,
                    depth_result.clear_direction, depth_result.region_distances_m,
                )

            if ssv2_sentence:
                logger.info(
                    "[%05d] %s risk=%.2f | det=%.2f wm=%.2f ta=%.2f | SSv2: %s (%.0f%%) | %s",
                    frame_idx, decision.action, decision.risk_score,
                    det_result.raw_risk, wm_risk, temporal_result.temporal_risk,
                    ssv2_sentence, ssv2_conf * 100, decision.explanation,
                )
            else:
                logger.info(
                    "[%05d] %s risk=%.2f | det=%.2f wm=%.2f ta=%.2f | %s",
                    frame_idx, decision.action, decision.risk_score,
                    det_result.raw_risk, wm_risk, temporal_result.temporal_risk,
                    decision.explanation,
                )

            # ── 7. Execute motor command ──────────────────────────────────────
            # ── 7b. Goal tracking + (Goal-Following mode) goal-directed steering ─
            goal_state = None
            if self._goal_tracker.active:
                goal_state = self._goal_tracker.update(
                    frame_bgr, depth_sampler=self._depth.depth_at_norm,
                )
                # Arrival: on the first frame the goal is reached, STOP and hold —
                # the robot waits for a new UI command (new goal / re-activate).
                if goal_state.reached and not self._goal_reached_handled:
                    self._goal_reached_handled = True
                    self.set_motor_enabled(False)
                    logger.info("Goal reached (≤ arrival distance) – stopping, awaiting UI command")
                # In Goal-Following mode, a lost goal stops the robot (don't wander).
                elif (self._goal_following and goal_state.lost
                        and not self._goal_lost_handled):
                    self._goal_lost_handled = True
                    self.set_motor_enabled(False)
                    logger.info("Goal lost while following – stopping, awaiting new goal")
                # Goal-Following: steer toward the goal when the path is clear
                # (safety/avoidance from decide() always overrides — see goal_steering).
                if self._goal_following and not goal_state.reached:
                    from goal_navigator import goal_steering
                    new_action, turn_dir = goal_steering(
                        decision.action, goal_state, self._goal_center_tol)
                    if new_action != decision.action:
                        from dataclasses import replace
                        decision = replace(
                            decision, action=new_action,
                            reroute_direction=turn_dir,
                            explanation=f"goal-follow → {new_action.value.lower()} "
                                        f"(bearing {goal_state.bearing_deg:+.0f}°)",
                        )
            with self._state_lock:
                motor_enabled = self._motor_enabled
                logging_enabled = self._logging_enabled
                self._goal_state = goal_state
            if motor_enabled:
                self._execute_action(self._robot, decision.action,
                                     getattr(decision, "reroute_direction", ""))

            # ── 8. Visualise ──────────────────────────────────────────────────
            annotated = self._visualizer.annotate(
                frame_bgr, det_result, decision, temporal_result, sonic_cm,
                ssv2_sentence=ssv2_sentence, depth=depth_result, goal=goal_state,
            )
            with self._annotated_lock:
                self._last_annotated_bgr = annotated
            self._visualizer.show(annotated)

            # ── 9. Log (only when the operator has logging enabled) ───────────
            if logging_enabled:
                lat["total"] = (time.monotonic() - proc_t0) * 1000.0
                net = self._cam_buf.get_net_stats()
                metrics = {
                    # inference latency (ms) — heavy models run every N frames, so
                    # their per-frame cost is near-0 on skipped frames and spikes on
                    # the compute tick; that periodicity is visible in the log.
                    "lat_total_ms":    lat.get("total", 0.0),
                    "lat_yolo_ms":     lat.get("yolo", 0.0),
                    "lat_wm_ms":       lat.get("wm", 0.0),
                    "lat_depth_ms":    lat.get("depth", 0.0),
                    "lat_temporal_ms": lat.get("temporal", 0.0),
                    "lat_ssv2_ms":     lat.get("ssv2", 0.0),
                    "lat_decision_ms": lat.get("decision", 0.0),
                    "reaction_ema_ms": self._reaction_ema * 1000.0,
                    # network / stream stats
                    "net_recv_fps":      net["recv_fps"],
                    "net_frame_bytes":   net["last_bytes"],
                    "net_frames_recv":   net["frames_recv"],
                    "net_frames_dropped": self._net_dropped_total,
                    "net_kbps":          net["kbps"],
                }
                # Raw (un-annotated) frame + depth for offline calibration from logs.
                depth_row = {
                    "center": regions.get("CENTER"),
                    "left": regions.get("LEFT"),
                    "right": regions.get("RIGHT"),
                } if depth_result.buffer_ready else None
                self._nav_logger.log_frame(
                    annotated, decision, det_result, sonic_cm,
                    ssv2_sentence=ssv2_sentence, metrics=metrics,
                    raw_frame=frame_bgr, depth=depth_row,
                )

            # ── 10. Broadcast AI status to TCP client ─────────────────────────
            self._broadcast_status(decision, temporal_result, sonic_cm,
                                   ssv2_sentence, depth_result, goal_state)

            # ── 11. Update shared state ───────────────────────────────────────
            with self._state_lock:
                self._state.action = decision.action
                self._state.risk_score = decision.risk_score
                self._state.world_model_label = wm_result.label
                self._state.temporal_pattern = temporal_result.pattern
                self._state.obstacles_detected = len(det_result.boxes)
                self._state.ultrasonic_cm = sonic_cm
                self._state.frame_count = frame_idx
                self._state.ssv2_sentence = ssv2_sentence
                self._state.ssv2_confidence = ssv2_conf

            # ── 12. Update the reaction-time estimate (governor input) ────────
            # EMA of the wall-clock time this frame took; the speed governor uses it
            # as the AI's reaction latency. The heavy models are off-loop now, so a
            # frame is fast; cap the sample so a rare one-off stall (a GC pause, a
            # cold op) can't balloon the EMA and wedge the governor into a long STOP
            # (as a 44 s SSv2 tick once pushed t_react to ~18 s).
            proc_dt = min(time.monotonic() - proc_t0, 0.5)
            self._reaction_ema = 0.4 * proc_dt + 0.6 * self._reaction_ema

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

        # All AI runs on the PC: YOLO always runs here, on the streamed frames
        # (live) or the demo video. The Pi is a thin client that only streams
        # frames and reports ultrasonic — it never runs YOLO.
        self._detector = Detector(cfg)
        try:
            self._detector.load()
            logger.info("YOLO11n loaded on the server (all AI runs on the PC)")
        except Exception as exc:
            logger.error(
                "YOLO failed to load on the server (%s) – detection disabled. "
                "Ensure 'ultralytics' is installed on the PC.", exc,
            )
            self._detector = None

        self._world_model = WorldModel(cfg)
        self._world_model.load()

        from ssv2_model import SSv2Recognizer
        self._ssv2 = SSv2Recognizer(cfg)
        self._ssv2.load()

        from depth_perception import DepthEstimator
        self._depth = DepthEstimator(cfg)
        self._depth.load()

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

    def _broadcast_status(self, decision, temporal_result, sonic_cm: float,
                          ssv2_sentence: str = "", depth_result=None,
                          goal_state=None) -> None:
        """
        Send a compact status string to all connected command clients so the
        lightweight client UI can display live AI state (in the panel below the
        video) without video decoding.

        Format:
          CMD_AISTATUS#<action>#<risk*100>#<wm_label>#<pattern>#<sonic_cm>#<ssv2>
                       #<clear_dist_m>#<clear_dir>#<goal_status>
                       #<depth_left_m>#<depth_right_m>#<goal_bearing_deg>#<goal_dist_m>\r\n
        goal_status ∈ none|tracking|lost|reached. The trailing depth_left/right +
        goal bearing/distance feed the 2D navigation map (-1 = unknown). Fields are
        appended, never reordered, so older clients that split on the first 6–10
        fields keep working.
        """
        if self._tcp_server is None:
            return
        try:
            # Action / MotionPattern are (str, Enum) members; on Python 3.11+
            # f"{member}" renders "Action.FORWARD" not "FORWARD", which breaks the
            # wire protocol and the UI colour lookups. Emit the plain .value.
            action = getattr(decision.action, "value", decision.action)
            wm_label = getattr(decision.world_model_label, "value", decision.world_model_label)
            pattern = getattr(temporal_result.pattern, "value", temporal_result.pattern)
            # '#' is the field separator; keep the sentence clean.
            ssv2 = (ssv2_sentence or "").replace("#", " ")
            # Depth free-space (shown in the UI panel below the video). -1 / "" when
            # the depth model isn't ready.
            # Per-side depth (LEFT/CENTER/RIGHT metres) for the 2D map; -1 = unknown.
            clear_left = clear_right = -1.0
            if depth_result is not None and getattr(depth_result, "buffer_ready", False):
                clear_dist = getattr(depth_result, "clear_distance_m", -1.0)
                clear_dir = getattr(depth_result, "clear_direction", "") or ""
                regions = getattr(depth_result, "region_distances_m", {}) or {}
                clear_left = regions.get("LEFT", -1.0)
                clear_right = regions.get("RIGHT", -1.0)
            else:
                clear_dist, clear_dir = -1.0, ""
            # Goal status for the UI ("Goal reached" banner etc.).
            if goal_state is None or not getattr(goal_state, "active", False):
                goal_status = "none"
            elif goal_state.reached:
                goal_status = "reached"
            elif goal_state.lost:
                goal_status = "lost"
            else:
                goal_status = "tracking"
            # Goal bearing (deg) + depth (m) so the map can place the goal marker.
            goal_bearing = float(getattr(goal_state, "bearing_deg", 0.0) or 0.0) if goal_state else 0.0
            goal_dist = getattr(goal_state, "distance_m", None) if goal_state else None
            goal_dist = goal_dist if (goal_dist is not None and goal_dist > 0) else -1.0
            msg = (
                f"CMD_AISTATUS#{action}"
                f"#{int(decision.risk_score * 100)}"
                f"#{wm_label}"
                f"#{pattern}"
                f"#{sonic_cm:.1f}"
                f"#{ssv2}"
                f"#{clear_dist:.2f}"
                f"#{clear_dir}"
                f"#{goal_status}"
                f"#{clear_left:.2f}"
                f"#{clear_right:.2f}"
                f"#{goal_bearing:.1f}"
                f"#{goal_dist:.2f}\r\n"
            )
            if self._tcp_server.isCmdServerConnected():
                self._tcp_server.sendDataToCmdClinet(msg)
        except Exception as exc:
            logger.debug("Status broadcast error: %s", exc)
