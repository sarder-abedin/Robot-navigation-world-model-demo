"""
main.py – Predictive Indoor Navigation System for the Freenove Tank Robot.

Entry point for both demo (recorded video) and live (physical robot) modes.

Usage:
  python main.py                                  # demo + predictive (defaults from config.yaml)
  python main.py --mode demo --nav predictive
  python main.py --mode demo --nav baseline
  python main.py --mode live  --nav predictive
  python main.py --video path/to/video.mp4        # override demo video path
  python main.py --build-anchors                  # calibrate V-JEPA 2 anchors

Press 'q' in the visualisation window to quit gracefully.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import cv2
import yaml

logger = logging.getLogger(__name__)


# ── Graceful shutdown flag ─────────────────────────────────────────────────────
_shutdown = False


def _handle_sigint(sig, frame):
    global _shutdown
    logger.warning("Interrupt received – shutting down…")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predictive Navigation Demo")
    p.add_argument("--config", default="config.yaml", help="Path to config file")
    p.add_argument(
        "--mode", choices=["demo", "live"], default=None,
        help="Override config mode: 'demo' (video file) or 'live' (physical robot)",
    )
    p.add_argument(
        "--nav", choices=["predictive", "baseline"], default=None,
        help="Navigation mode: 'predictive' uses V-JEPA 2, 'baseline' is reactive only",
    )
    p.add_argument("--video", default=None, help="Path to demo video (overrides config)")
    p.add_argument(
        "--build-anchors", action="store_true",
        help="Interactively build V-JEPA 2 anchors from live camera frames and exit",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV display window (useful for headless runs)",
    )
    return p.parse_args()


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Anchor-building utility ────────────────────────────────────────────────────

def run_anchor_builder(cfg: dict) -> None:
    """
    Collect example frames interactively and build V-JEPA 2 anchors.

    Press 'o' to label current frame as OBSTACLE.
    Press 'c' to label current frame as CLEAR.
    Press 'q' to finish and save anchors.
    """
    from src.camera import build_camera
    from src.world_model import WorldModel

    cam = build_camera(cfg)
    cam.open()
    wm = WorldModel(cfg)
    wm.load()

    obs_frames, clear_frames = [], []
    print("\n[Anchor Builder]")
    print("  'o' → label frame as OBSTACLE")
    print("  'c' → label frame as CLEAR")
    print("  'q' → finish and save\n")

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        display = frame.copy()
        cv2.putText(display,
                    f"obs={len(obs_frames)} clear={len(clear_frames)} | o/c/q",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Anchor Builder", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("o"):
            obs_frames.append(frame)
            print(f"  OBSTACLE frame saved ({len(obs_frames)} total)")
        elif key == ord("c"):
            clear_frames.append(frame)
            print(f"  CLEAR frame saved ({len(clear_frames)} total)")
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if obs_frames and clear_frames:
        wm.build_anchors(obs_frames, clear_frames)
        print("Anchors updated successfully.")
    else:
        print("Not enough frames collected – anchors not updated.")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(cfg: dict, nav_mode: str) -> None:
    from src.camera import build_camera
    from src.decision import DecisionFuser
    from src.detector import Detector
    from src.logger import NavigationLogger
    from src.robot import build_robot, execute_action
    from src.temporal_action import TemporalActionRecognizer, detection_to_state
    from src.visualization import Visualizer
    from src.world_model import WorldModel

    # ── Initialise all components ─────────────────────────────────────────────
    camera = build_camera(cfg)
    detector = Detector(cfg)
    world_model = WorldModel(cfg)
    temporal = TemporalActionRecognizer(cfg)
    fuser = DecisionFuser(cfg, nav_mode)
    robot = build_robot(cfg)
    nav_logger = NavigationLogger(cfg, nav_mode)
    visualizer = Visualizer(cfg, nav_mode)

    logger.info("=== Predictive Navigation System starting ===")
    logger.info("Mode: %s | Nav: %s", cfg.get("mode", "demo"), nav_mode)

    camera.open()
    detector.load()
    world_model.load()
    robot.connect()

    # ── Main processing loop ──────────────────────────────────────────────────
    frame_idx = 0
    try:
        while not _shutdown:
            ok, frame = camera.read()
            if not ok:
                logger.warning("Frame read failed – exiting loop")
                break

            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── 1. Object detection ───────────────────────────────────────────
            det_result = detector.detect(frame)

            # ── 2. V-JEPA 2 world model prediction ───────────────────────────
            world_model.push_frame(rgb)
            wm_result = world_model.predict()

            # Use world-model risk if buffer is ready; else fall back to detector risk
            wm_risk = wm_result.predicted_risk if wm_result.buffer_ready else det_result.raw_risk

            # ── 3. Temporal motion pattern recognition ────────────────────────
            obs_state = detection_to_state(det_result)
            temporal.push(obs_state)
            temporal_result = temporal.classify()

            # ── 4. Decision fusion ────────────────────────────────────────────
            decision = fuser.decide(
                detector_risk=det_result.raw_risk,
                world_model_risk=wm_risk,
                temporal_risk=temporal_result.temporal_risk,
                world_model_label=wm_result.label,
                temporal_pattern=temporal_result.pattern,
            )

            logger.info(
                "[Frame %05d] action=%s risk=%.2f | det=%.2f wm=%.2f ta=%.2f | %s",
                frame_idx,
                decision.action,
                decision.risk_score,
                det_result.raw_risk,
                wm_risk,
                temporal_result.temporal_risk,
                decision.explanation,
            )

            # ── 5. Execute robot command ──────────────────────────────────────
            execute_action(robot, decision.action)

            # ── 6. Visualise ──────────────────────────────────────────────────
            annotated = visualizer.annotate(frame, det_result, decision, temporal_result)

            # ── 7. Log ────────────────────────────────────────────────────────
            nav_logger.log_frame(annotated, decision, det_result)

            # ── 8. Display (quit on 'q') ───────────────────────────────────────
            if not visualizer.show(annotated):
                logger.info("User requested quit via 'q'")
                break

    except Exception as exc:
        logger.exception("Unhandled exception in main loop: %s", exc)
        robot.safe_stop()
    finally:
        robot.stop()
        robot.disconnect()
        camera.release()
        nav_logger.close()
        visualizer.close()
        logger.info("=== Shutdown complete ===")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.mode:
        cfg["mode"] = args.mode
    if args.video:
        cfg["camera"]["demo_video_path"] = args.video
    if args.no_display:
        cfg["visualization"]["show_window"] = False

    nav_mode = args.nav or cfg.get("navigation_mode", "predictive")

    if args.build_anchors:
        run_anchor_builder(cfg)
        return

    run(cfg, nav_mode)


if __name__ == "__main__":
    main()
