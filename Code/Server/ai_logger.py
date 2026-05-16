"""
ai_logger.py – Structured logging of the full AI pipeline output.

Every processed frame produces one CSV row capturing:
  timestamp, frame_idx, navigation_mode, action, fused_risk,
  detector_risk, world_model_risk, temporal_risk,
  world_model_label, temporal_pattern,
  obstacles_detected, obstacle_in_center, closest_area,
  ultrasonic_cm, explanation

Annotated JPEG frames are saved every N frames to the run directory.
A system.log text file captures all Python logging output.

Usage:
  nav_log = NavigationLogger(cfg, nav_mode)
  nav_log.log_frame(annotated_bgr_frame, decision_result, det_result, sonic_cm)
  nav_log.close()
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class NavigationLogger:
    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        log_cfg = cfg["logging"]
        self._log_dir = Path(log_cfg["log_dir"])
        self._save_frames = log_cfg.get("save_annotated_frames", True)
        self._frame_interval = log_cfg.get("annotated_frame_interval", 5)
        self._csv_enabled = log_cfg.get("csv_log", True)
        self._nav_mode = navigation_mode

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._log_dir / f"run_{run_ts}_{navigation_mode}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        if self._save_frames:
            self._frames_dir = self._run_dir / "frames"
            self._frames_dir.mkdir(exist_ok=True)

        self._csv_path = self._run_dir / "navigation_log.csv"
        self._csv_file = None
        self._csv_writer = None
        self._frame_idx = 0

        self._setup_logging(log_cfg["log_level"])
        self._open_csv()
        logger.info("NavigationLogger → %s", self._run_dir)

    def log_frame(
        self,
        annotated_frame: np.ndarray,
        decision_result,
        detector_result,
        ultrasonic_cm: float = -1.0,
    ) -> None:
        ts = time.time()
        self._frame_idx += 1

        if self._csv_writer:
            self._csv_writer.writerow({
                "timestamp":       f"{ts:.4f}",
                "frame_idx":       self._frame_idx,
                "nav_mode":        self._nav_mode,
                "action":          decision_result.action,
                "risk_score":      f"{decision_result.risk_score:.4f}",
                "detector_risk":   f"{decision_result.detector_risk:.4f}",
                "world_model_risk":f"{decision_result.world_model_risk:.4f}",
                "temporal_risk":   f"{decision_result.temporal_risk:.4f}",
                "wm_label":        decision_result.world_model_label,
                "temporal_pattern":decision_result.temporal_pattern,
                "obstacles":       len(detector_result.boxes),
                "in_center":       int(detector_result.obstacle_in_center),
                "closest_area":    f"{detector_result.closest_area:.4f}",
                "ultrasonic_cm":   f"{ultrasonic_cm:.1f}",
                "explanation":     decision_result.explanation,
            })
            if self._frame_idx % 20 == 0:
                self._csv_file.flush()

        if self._save_frames and (self._frame_idx % self._frame_interval == 0):
            fname = self._frames_dir / f"frame_{self._frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), annotated_frame)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        logger.info("Logger closed → %s", self._run_dir)

    # ── Private ───────────────────────────────────────────────────────────────

    def _open_csv(self) -> None:
        if not self._csv_enabled:
            return
        fieldnames = [
            "timestamp", "frame_idx", "nav_mode",
            "action", "risk_score",
            "detector_risk", "world_model_risk", "temporal_risk",
            "wm_label", "temporal_pattern",
            "obstacles", "in_center", "closest_area",
            "ultrasonic_cm", "explanation",
        ]
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def _setup_logging(self, level_str: str) -> None:
        level = getattr(logging, level_str.upper(), logging.INFO)
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            ))
            root.addHandler(h)
        fh = logging.FileHandler(self._run_dir / "system.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root.addHandler(fh)
        root.setLevel(level)
