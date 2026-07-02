"""
detector_robot.py – Lightweight YOLOv8n detector for the Raspberry Pi client.

Runs inference locally on the Pi so only aggregated detection results are sent
to the PC server via CMD_DETECTION, reducing bandwidth and removing GPU-heavy
YOLO work from the PC.

Output: DetectionPacket with:
  yolo_risk_pct  – heuristic obstacle risk 0-100
  obs_in_center  – bool, largest obstacle is in the centre danger zone
  area_frac_pct  – largest bbox area as % of frame area (0-100)
  centroid_x_pct – horizontal centroid of the closest obstacle (0-100)
  n_obstacles    – total obstacle count
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionPacket:
    yolo_risk_pct: int = 0       # 0-100
    obs_in_center: bool = False
    area_frac_pct: int = 0       # 0-100
    centroid_x_pct: int = 50     # 0-100  (50 = centre)
    n_obstacles: int = 0


class DetectorRobot:
    """YOLOv8n obstacle detector for the Raspberry Pi client."""

    def __init__(self, cfg: dict):
        det = cfg.get("detector", {})
        self._model_name = det.get("model", "yolov8n.pt")
        self._conf = det.get("conf", 0.35)
        self._center_zone_width = det.get("center_zone_width", 0.40)
        self._run_every = det.get("run_every_n_frames", 2)
        self._model = None
        self._frame_count = 0
        self._last = DetectionPacket()

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore
        self._model = YOLO(self._model_name)
        logger.info("DetectorRobot loaded: %s", self._model_name)

    def detect(self, frame_bgr) -> DetectionPacket:
        """
        Run YOLOv8n on a BGR numpy frame.
        Returns cached result on skipped frames to avoid stale data.
        """
        self._frame_count += 1
        if self._frame_count % self._run_every != 0 or self._model is None:
            return self._last

        h, w = frame_bgr.shape[:2]
        cx_lo = w * (0.5 - self._center_zone_width / 2)
        cx_hi = w * (0.5 + self._center_zone_width / 2)

        results = self._model(frame_bgr, conf=self._conf, verbose=False)[0]

        obs_in_center = False
        best_area = 0.0
        best_cx = 0.5
        n = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1) / (w * h)
            if area > best_area:
                best_area = area
                best_cx = (x1 + x2) / 2 / w
            if cx_lo <= (x1 + x2) / 2 <= cx_hi:
                obs_in_center = True
            n += 1

        # Heuristic risk mirrors the PC-side Detector._compute_risk()
        if n == 0:
            risk = 0.0
        else:
            center_penalty = 0.40 if obs_in_center else 0.0
            area_penalty = min(best_area / 0.08, 1.0) * 0.40
            count_penalty = min(n / 3, 1.0) * 0.20
            risk = min(center_penalty + area_penalty + count_penalty, 1.0)

        self._last = DetectionPacket(
            yolo_risk_pct=int(risk * 100),
            obs_in_center=obs_in_center,
            area_frac_pct=int(best_area * 100),
            centroid_x_pct=int(best_cx * 100),
            n_obstacles=n,
        )
        return self._last
