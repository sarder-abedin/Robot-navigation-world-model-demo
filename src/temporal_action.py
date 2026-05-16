"""
temporal_action.py – SSv2-style temporal motion pattern recognition.

Motivation
──────────
Something-Something V2 (SSv2) is a video dataset of fine-grained hand/object
interactions described with motion verbs ("something moving towards camera",
"something covering something", etc.).  We adopt the same *temporal
reasoning philosophy*: instead of classifying single frames, we classify
the *sequence of changes* in the obstacle state over the last N frames.

This lets us distinguish:
  APPROACHING  – obstacle is getting bigger / moving into center over time
  CROSSING     – obstacle crosses the frame L→R or R→L (transient blockage)
  BLOCKING     – obstacle is large, centered, and not moving (static wall)
  CLEARING     – obstacle was present but is now shrinking / moving away
  STATIC_CLEAR – no obstacle was ever detected in the window
  UNCERTAIN    – not enough signal to classify

Each pattern maps to a temporal_risk modifier that decision.py uses.

TemporalState (per frame):
  obstacle_present  bool
  in_center         bool
  area_frac         float
  centroid_x        float   (normalised 0-1, left to right)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class MotionPattern(str, Enum):
    APPROACHING = "APPROACHING"
    CROSSING = "CROSSING"
    BLOCKING = "BLOCKING"
    CLEARING = "CLEARING"
    STATIC_CLEAR = "STATIC_CLEAR"
    UNCERTAIN = "UNCERTAIN"


# Risk modifier for each pattern (additive bonus on top of base risk)
PATTERN_RISK: dict[MotionPattern, float] = {
    MotionPattern.APPROACHING: 0.70,
    MotionPattern.BLOCKING: 0.85,
    MotionPattern.CROSSING: 0.45,
    MotionPattern.CLEARING: 0.10,
    MotionPattern.STATIC_CLEAR: 0.0,
    MotionPattern.UNCERTAIN: 0.25,
}


@dataclass
class FrameObstacleState:
    obstacle_present: bool
    in_center: bool
    area_frac: float
    centroid_x: float  # 0.0 = left edge, 1.0 = right edge


@dataclass
class TemporalResult:
    pattern: MotionPattern
    temporal_risk: float   # in [0, 1]
    description: str


class TemporalActionRecognizer:
    """
    Classifies obstacle motion patterns from a rolling window of per-frame
    obstacle states (not raw pixels) so it runs with zero GPU cost.

    The recognition logic is deliberately rule-based, mirroring how SSv2
    categories are labelled: each label corresponds to a specific directional
    or size-change trajectory.
    """

    def __init__(self, cfg: dict):
        ta_cfg = cfg["temporal_action"]
        self._window = ta_cfg["window_size"]
        self._approach_ratio = ta_cfg["approach_ratio"]
        self._clear_ratio = ta_cfg["clear_ratio"]
        self._speed_thresh = ta_cfg["movement_speed_threshold"]

        self._history: deque[FrameObstacleState] = deque(maxlen=self._window)

    def push(self, state: FrameObstacleState) -> None:
        self._history.append(state)

    def classify(self) -> TemporalResult:
        if len(self._history) < 3:
            return TemporalResult(
                pattern=MotionPattern.UNCERTAIN,
                temporal_risk=PATTERN_RISK[MotionPattern.UNCERTAIN],
                description="Not enough history",
            )

        hist = list(self._history)
        n = len(hist)

        present_mask = [s.obstacle_present for s in hist]
        present_ratio = sum(present_mask) / n
        center_mask = [s.in_center for s in hist if s.obstacle_present]
        center_ratio = sum(center_mask) / len(center_mask) if center_mask else 0.0

        areas = [s.area_frac for s in hist]
        cx_vals = [s.centroid_x for s in hist if s.obstacle_present]

        # ── STATIC_CLEAR: nothing detected for the whole window ───────────────
        if present_ratio < (1 - self._clear_ratio):
            return TemporalResult(
                pattern=MotionPattern.STATIC_CLEAR,
                temporal_risk=PATTERN_RISK[MotionPattern.STATIC_CLEAR],
                description="Path clear",
            )

        # ── CLEARING: obstacle was present early but absent recently ──────────
        early_present = sum(present_mask[: n // 2]) / (n // 2)
        late_present = sum(present_mask[n // 2 :]) / (n - n // 2)
        if early_present > 0.5 and late_present < (1 - self._clear_ratio):
            return TemporalResult(
                pattern=MotionPattern.CLEARING,
                temporal_risk=PATTERN_RISK[MotionPattern.CLEARING],
                description="Obstacle clearing from path",
            )

        # ── APPROACHING: area growing AND obstacle in center ──────────────────
        if len(areas) >= 4 and center_ratio > self._approach_ratio:
            area_trend = _linear_trend(areas)
            if area_trend > 0.003:  # growing by >0.3 % per frame
                return TemporalResult(
                    pattern=MotionPattern.APPROACHING,
                    temporal_risk=PATTERN_RISK[MotionPattern.APPROACHING],
                    description=f"Obstacle approaching (area trend +{area_trend:.4f}/frame)",
                )

        # ── CROSSING: obstacle centroid moves laterally, not towards camera ───
        if len(cx_vals) >= 4:
            cx_trend = _linear_trend(cx_vals)
            area_trend = _linear_trend(areas[-len(cx_vals):])
            lateral_speed = abs(cx_trend)
            if lateral_speed > 0.02 and abs(area_trend) < 0.003:
                direction = "left→right" if cx_trend > 0 else "right→left"
                return TemporalResult(
                    pattern=MotionPattern.CROSSING,
                    temporal_risk=PATTERN_RISK[MotionPattern.CROSSING],
                    description=f"Obstacle crossing ({direction})",
                )

        # ── BLOCKING: large, centered, mostly stationary ──────────────────────
        recent_areas = areas[max(0, n - 5):]
        mean_area = float(np.mean(recent_areas)) if recent_areas else 0.0
        if present_ratio > 0.7 and center_ratio > 0.5 and mean_area > 0.05:
            return TemporalResult(
                pattern=MotionPattern.BLOCKING,
                temporal_risk=PATTERN_RISK[MotionPattern.BLOCKING],
                description=f"Static obstacle blocking path (area={mean_area:.2f})",
            )

        return TemporalResult(
            pattern=MotionPattern.UNCERTAIN,
            temporal_risk=PATTERN_RISK[MotionPattern.UNCERTAIN],
            description="Ambiguous motion pattern",
        )


def detection_to_state(det_result) -> FrameObstacleState:
    """
    Convert a DetectionResult (from detector.py) into a FrameObstacleState
    for the temporal recogniser.

    centroid_x is derived from the largest obstacle bounding box.
    """
    if not det_result.boxes:
        return FrameObstacleState(
            obstacle_present=False,
            in_center=False,
            area_frac=0.0,
            centroid_x=0.5,
        )

    # Pick the largest obstacle box
    largest_idx = int(np.argmax([
        (x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in det_result.boxes
    ]))
    x1, y1, x2, y2 = det_result.boxes[largest_idx]
    cx = ((x1 + x2) / 2)

    # We need frame width to normalise; store raw pixel centroid and let the
    # recogniser work with ratios.  We normalise assuming 640 px default width.
    # (The detector does not carry frame size; this is a reasonable assumption.)
    ASSUMED_WIDTH = 640
    cx_norm = float(np.clip(cx / ASSUMED_WIDTH, 0.0, 1.0))

    return FrameObstacleState(
        obstacle_present=True,
        in_center=det_result.obstacle_in_center,
        area_frac=det_result.closest_area,
        centroid_x=cx_norm,
    )


def _linear_trend(values: list[float]) -> float:
    """Slope of a least-squares line fit through the values list."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])
