"""
detector.py – Per-frame object detection using YOLO11.

Outputs:
  DetectionResult
    .boxes        list of (x1,y1,x2,y2) in pixel coords
    .labels       list of class name strings
    .confidences  list of float scores
    .obstacle_in_center  bool – at least one obstacle occupies the center zone
    .closest_area        float – largest obstacle bbox area as fraction of frame
    .raw_risk            float in [0,1] – a quick heuristic risk estimate
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import so the module can be imported without ultralytics installed
_yolo_model = None


@dataclass
class DetectionResult:
    boxes: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    confidences: list = field(default_factory=list)
    obstacle_in_center: bool = False
    closest_area: float = 0.0
    raw_risk: float = 0.0


class Detector:
    """
    Thin wrapper around a YOLO11 model.

    Only classes listed in config obstacle_classes are treated as obstacles.
    The detector also marks whether any obstacle sits in the center horizontal
    band of the frame, because that region is most relevant for forward motion.
    """

    def __init__(self, cfg: dict):
        det_cfg = cfg["detector"]
        self._model_name = det_cfg["model"]
        self._conf = det_cfg["confidence_threshold"]
        self._iou = det_cfg["iou_threshold"]
        self._obstacle_classes = set(det_cfg["obstacle_classes"])
        self._center_ratio = det_cfg["center_zone_ratio"]
        self._close_area_thresh = det_cfg["close_area_threshold"]
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore
        self._model = YOLO(self._model_name)
        logger.info("YOLO11 model loaded: %s", self._model_name)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("Detector not loaded – call load() first")

        h, w = frame.shape[:2]
        center_x_lo = w * (0.5 - self._center_ratio / 2)
        center_x_hi = w * (0.5 + self._center_ratio / 2)

        results = self._model(
            frame,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )[0]

        boxes, labels, confidences = [], [], []
        obstacle_in_center = False
        closest_area = 0.0

        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            if cls_name not in self._obstacle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2))
            labels.append(cls_name)
            confidences.append(conf)

            # Area fraction relative to frame
            area_frac = (x2 - x1) * (y2 - y1) / (w * h)
            closest_area = max(closest_area, area_frac)

            # Check center-zone overlap
            box_cx = (x1 + x2) / 2
            if center_x_lo <= box_cx <= center_x_hi:
                obstacle_in_center = True

        raw_risk = self._compute_raw_risk(
            obstacle_in_center, closest_area, len(boxes)
        )

        return DetectionResult(
            boxes=boxes,
            labels=labels,
            confidences=confidences,
            obstacle_in_center=obstacle_in_center,
            closest_area=closest_area,
            raw_risk=raw_risk,
        )

    def _compute_raw_risk(
        self,
        in_center: bool,
        area: float,
        count: int,
    ) -> float:
        """
        Heuristic risk score in [0,1] derived from current-frame detections.

        This is the *reactive* signal. The world model provides the
        *predictive* signal; decision.py fuses both.
        """
        if count == 0:
            return 0.0
        center_penalty = 0.4 if in_center else 0.0
        area_penalty = min(area / self._close_area_thresh, 1.0) * 0.4
        count_penalty = min(count / 3, 1.0) * 0.2
        return min(center_penalty + area_penalty + count_penalty, 1.0)
