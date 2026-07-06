"""
detector.py – Per-frame object detection (YOLOv8) for the Raspberry Pi server.

Only obstacle-class detections are returned.  The output is a DetectionResult
that also carries a quick heuristic risk score suitable for the decision fuser.

The detector runs every N frames (configurable via detector.run_every_n_frames)
to keep CPU usage manageable on a Raspberry Pi 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    boxes: list = field(default_factory=list)        # [(x1,y1,x2,y2), …] pixel coords
    labels: list = field(default_factory=list)       # class name strings
    confidences: list = field(default_factory=list)  # float scores
    obstacle_in_center: bool = False
    closest_area: float = 0.0   # largest obstacle bbox as fraction of frame area
    closest_label: str = ""     # YOLO class of the largest/closest obstacle (SSv2 filler)
    raw_risk: float = 0.0       # heuristic reactive risk in [0,1]
    frame_width: int = 0
    frame_height: int = 0


class Detector:
    """
    Thin YOLOv8 wrapper.

    In the predictive pipeline the detector provides the *instantaneous* risk
    signal.  V-JEPA 2 provides the *future* risk signal.  Decision.py fuses both.
    """

    def __init__(self, cfg: dict):
        det_cfg = cfg["detector"]
        self._model_name = det_cfg["model"]
        self._conf = det_cfg["confidence_threshold"]
        self._iou = det_cfg["iou_threshold"]
        self._obstacle_classes = set(det_cfg["obstacle_classes"])
        self._center_ratio = det_cfg["center_zone_ratio"]
        self._close_area_thresh = max(1e-6, float(det_cfg["close_area_threshold"]))
        self._run_every = det_cfg.get("run_every_n_frames", 2)
        self._model = None
        self._frame_count = 0
        self._last_result: DetectionResult = DetectionResult()

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore
        self._model = YOLO(self._model_name)
        logger.info("YOLOv8 loaded: %s", self._model_name)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a BGR uint8 numpy frame (OpenCV order — ultralytics
        assumes numpy input is BGR and flips it to RGB internally).

        Returns the cached last result on skipped frames to avoid stale
        processing without wasting CPU cycles.
        """
        self._frame_count += 1
        if self._frame_count % self._run_every != 0:
            return self._last_result

        if self._model is None:
            raise RuntimeError("Detector not loaded – call load() first")

        h, w = frame.shape[:2]
        cx_lo = w * (0.5 - self._center_ratio / 2)
        cx_hi = w * (0.5 + self._center_ratio / 2)

        # ultralytics takes numpy frames in BGR (see docstring); pass frame_bgr.
        results = self._model(frame, conf=self._conf, iou=self._iou, verbose=False)[0]

        boxes, labels, confidences = [], [], []
        obstacle_in_center = False
        closest_area = 0.0
        closest_label = ""   # label of the largest/closest obstacle (SSv2 filler)

        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            if cls_name not in self._obstacle_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2))
            labels.append(cls_name)
            confidences.append(conf)

            area_frac = (x2 - x1) * (y2 - y1) / (w * h)
            # Track which object is the largest (closest), not just the max area,
            # so the SSv2 "something" slot is filled with the right YOLO class.
            if area_frac > closest_area:
                closest_area = area_frac
                closest_label = cls_name

            box_cx = (x1 + x2) / 2
            if cx_lo <= box_cx <= cx_hi:
                obstacle_in_center = True

        raw_risk = self._compute_risk(obstacle_in_center, closest_area, len(boxes))
        self._last_result = DetectionResult(
            boxes=boxes,
            labels=labels,
            confidences=confidences,
            obstacle_in_center=obstacle_in_center,
            closest_area=closest_area,
            closest_label=closest_label,
            raw_risk=raw_risk,
            frame_width=w,
            frame_height=h,
        )
        return self._last_result

    def _compute_risk(self, in_center: bool, area: float, count: int) -> float:
        if count == 0:
            return 0.0
        center_penalty = 0.40 if in_center else 0.0
        area_penalty = min(area / self._close_area_thresh, 1.0) * 0.40
        count_penalty = min(count / 3, 1.0) * 0.20
        return min(center_penalty + area_penalty + count_penalty, 1.0)
