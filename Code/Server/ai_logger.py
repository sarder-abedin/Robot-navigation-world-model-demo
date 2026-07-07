"""
ai_logger.py – Structured logging of the full AI pipeline output.

Every processed frame produces one CSV row capturing:
  timestamp, frame_idx, navigation_mode, action, fused_risk,
  detector_risk, world_model_risk, temporal_risk,
  world_model_label, temporal_pattern,
  obstacles_detected, obstacle_in_center, closest_area,
  ultrasonic_cm, ssv2, explanation,
  # per-frame inference latency (ms): total + each stage
  lat_total_ms, lat_yolo_ms, lat_wm_ms, lat_depth_ms,
  lat_temporal_ms, lat_ssv2_ms, lat_decision_ms, reaction_ema_ms,
  # camera-stream network statistics
  net_recv_fps, net_frame_bytes, net_frames_recv, net_frames_dropped, net_kbps

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

# Per-frame inference-latency + network-statistics columns (appended to the CSV
# after the core nav columns). Kept in one place so the header and each row stay
# in sync. Values default to 0 when the pipeline doesn't supply metrics.
METRIC_FIELDS = [
    "lat_total_ms", "lat_yolo_ms", "lat_wm_ms", "lat_depth_ms",
    "lat_temporal_ms", "lat_ssv2_ms", "lat_decision_ms", "reaction_ema_ms",
    "net_recv_fps", "net_frame_bytes", "net_frames_recv",
    "net_frames_dropped", "net_kbps",
]


def _fmt_depth(v) -> str:
    """Format a depth value (m) for the CSV; -1 when unknown/None."""
    try:
        return f"{float(v):.3f}" if v is not None and float(v) > 0 else "-1"
    except (TypeError, ValueError):
        return "-1"


def _metric_row(metrics: dict | None) -> dict:
    """Format the metric columns for one CSV row (0 when absent)."""
    metrics = metrics or {}
    row = {}
    for k in METRIC_FIELDS:
        v = metrics.get(k, 0)
        # byte / frame counts are integers; latencies / rates get 2 decimals
        row[k] = str(int(v)) if k in (
            "net_frame_bytes", "net_frames_recv", "net_frames_dropped"
        ) else f"{float(v):.2f}"
    return row


class NavigationLogger:
    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        log_cfg = cfg["logging"]
        self._log_dir = Path(log_cfg["log_dir"])
        self._save_frames = log_cfg.get("save_annotated_frames", True)
        self._frame_interval = log_cfg.get("annotated_frame_interval", 5)
        # Raw (un-annotated) frames for offline V-JEPA 2 anchor calibration. Kept
        # in a SEPARATE raw_frames/ folder so the anchor tool never ingests HUD
        # overlays. Off by default (extra disk); enable before a run you plan to
        # calibrate from. See CALIBRATION.md / calibrate_from_logs.py.
        self._save_raw = log_cfg.get("save_raw_frames", False)
        self._raw_interval = int(log_cfg.get("raw_frame_interval", self._frame_interval))
        self._csv_enabled = log_cfg.get("csv_log", True)
        self._nav_mode = navigation_mode

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._log_dir / f"run_{run_ts}_{navigation_mode}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        if self._save_frames:
            self._frames_dir = self._run_dir / "frames"
            self._frames_dir.mkdir(exist_ok=True)
        if self._save_raw:
            self._raw_dir = self._run_dir / "raw_frames"
            self._raw_dir.mkdir(exist_ok=True)

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
        ssv2_sentence: str = "",
        metrics: dict | None = None,
        raw_frame: np.ndarray | None = None,
        depth: dict | None = None,
    ) -> None:
        """Legacy full-pipeline logging (kept for backward compat with tests).

        `metrics` optionally carries per-frame inference latencies (ms) and
        network/stream statistics (see METRIC_FIELDS); absent → logged as 0.
        `raw_frame` (BGR, un-annotated) is saved to raw_frames/ when save_raw_frames
        is on — for offline anchor calibration. `depth` = {"center","left","right"}
        metres, logged so depth scale can be calibrated from the logs.
        """
        ts = time.time()
        self._frame_idx += 1
        depth = depth or {}

        if self._csv_writer:
            row = {
                "timestamp":       f"{ts:.4f}",
                "frame_idx":       self._frame_idx,
                "nav_mode":        self._nav_mode,
                "action":          getattr(decision_result.action, "value", decision_result.action),
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
                "ssv2":            ssv2_sentence,
                "explanation":     decision_result.explanation,
                "depth_center_m":  _fmt_depth(depth.get("center")),
                "depth_left_m":    _fmt_depth(depth.get("left")),
                "depth_right_m":   _fmt_depth(depth.get("right")),
            }
            row.update(_metric_row(metrics))
            self._csv_writer.writerow(row)
            if self._frame_idx % 20 == 0:
                self._csv_file.flush()

        if self._save_frames and (self._frame_idx % self._frame_interval == 0):
            fname = self._frames_dir / f"frame_{self._frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), annotated_frame)

        if self._save_raw and raw_frame is not None and (self._frame_idx % self._raw_interval == 0):
            fname = self._raw_dir / f"frame_{self._frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), raw_frame)

    def log_detection_frame(
        self,
        annotated_frame: np.ndarray,
        detector_result,
        ultrasonic_cm: float = -1.0,
        client_action: str = "UNKNOWN",
    ) -> None:
        """
        Pi-side detection logging (post-refactor).

        Records YOLO11 detection data and the last action received from the
        client PC.  V-JEPA 2 / temporal / decision columns are omitted because
        those computations now run on the client.
        """
        ts = time.time()
        self._frame_idx += 1

        if self._csv_writer:
            self._csv_writer.writerow({
                "timestamp":       f"{ts:.4f}",
                "frame_idx":       self._frame_idx,
                "nav_mode":        self._nav_mode,
                "action":          client_action,
                "risk_score":      f"{detector_result.raw_risk:.4f}",
                "detector_risk":   f"{detector_result.raw_risk:.4f}",
                "world_model_risk":"client",
                "temporal_risk":   "client",
                "wm_label":        "client",
                "temporal_pattern":"client",
                "obstacles":       len(detector_result.boxes),
                "in_center":       int(detector_result.obstacle_in_center),
                "closest_area":    f"{detector_result.closest_area:.4f}",
                "ultrasonic_cm":   f"{ultrasonic_cm:.1f}",
                "explanation":     "pi-side detection only",
            })
            if self._frame_idx % 20 == 0:
                self._csv_file.flush()

        if self._save_frames and (self._frame_idx % self._frame_interval == 0):
            fname = self._frames_dir / f"frame_{self._frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), annotated_frame)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        # Detach the per-run FileHandler so repeated NavigationLogger instances
        # (tests, pipeline restarts) don't accumulate handlers / leak fds and
        # tee every record into every prior run's system.log.
        if getattr(self, "_file_handler", None) is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
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
            "ultrasonic_cm", "ssv2", "explanation",
            "depth_center_m", "depth_left_m", "depth_right_m",
            *METRIC_FIELDS,
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
        self._file_handler = fh  # kept so close() can detach it
        root.setLevel(level)
