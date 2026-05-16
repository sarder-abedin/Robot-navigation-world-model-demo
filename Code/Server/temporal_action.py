"""
temporal_action.py – Something-Something V2-style motion pattern recognition.

Philosophy
──────────
SSv2 labels videos with motion-centric phrases:
  "something moving towards camera"
  "something crossing left-to-right"
  "something blocking view"
  etc.

We apply the same temporal reasoning philosophy to the robot's obstacle history:
instead of classifying a single frame, we classify the *trajectory* of obstacle
states over the last N frames.

This gives the robot a language of anticipation:
  APPROACHING → the thing will block us if we keep going
  CROSSING    → transient; wait briefly and path will clear
  BLOCKING    → large, centred, static — stop or reroute
  CLEARING    → was present, now going away
  STATIC_CLEAR→ never there in this window
  UNCERTAIN   → ambiguous

Each pattern maps to a temporal_risk value that feeds into the decision fuser.
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


PATTERN_RISK: dict[MotionPattern, float] = {
    MotionPattern.APPROACHING:   0.70,
    MotionPattern.BLOCKING:      0.85,
    MotionPattern.CROSSING:      0.45,
    MotionPattern.CLEARING:      0.10,
    MotionPattern.STATIC_CLEAR:  0.00,
    MotionPattern.UNCERTAIN:     0.25,
}


@dataclass
class FrameObstacleState:
    obstacle_present: bool
    in_center: bool
    area_frac: float          # bbox area as fraction of frame
    centroid_x: float         # 0.0 = left, 1.0 = right (normalised)


@dataclass
class TemporalResult:
    pattern: MotionPattern
    temporal_risk: float
    description: str


class TemporalActionRecognizer:
    """
    Classifies the motion pattern of obstacles over a rolling window.

    Rule-based (no GPU) — runs every frame without computational cost.
    """

    def __init__(self, cfg: dict):
        ta_cfg = cfg["temporal_action"]
        self._window = ta_cfg["window_size"]
        self._approach_ratio = ta_cfg["approach_ratio"]
        self._clear_ratio = ta_cfg["clear_ratio"]
        self._history: deque[FrameObstacleState] = deque(maxlen=self._window)

    def push(self, state: FrameObstacleState) -> None:
        self._history.append(state)

    def classify(self) -> TemporalResult:
        if len(self._history) < 3:
            return _result(MotionPattern.UNCERTAIN, "Not enough history")

        hist = list(self._history)
        n = len(hist)

        present_mask = [s.obstacle_present for s in hist]
        present_ratio = sum(present_mask) / n

        present_frames = [s for s in hist if s.obstacle_present]
        center_ratio = (
            sum(s.in_center for s in present_frames) / len(present_frames)
            if present_frames else 0.0
        )
        areas = [s.area_frac for s in hist]
        cx_vals = [s.centroid_x for s in present_frames] if present_frames else []

        # ── STATIC_CLEAR ──────────────────────────────────────────────────────
        if present_ratio < (1 - self._clear_ratio):
            return _result(MotionPattern.STATIC_CLEAR, "Path clear")

        # ── CLEARING: present early, absent recently ──────────────────────────
        early = sum(present_mask[: n // 2]) / (n // 2)
        late = sum(present_mask[n // 2 :]) / (n - n // 2)
        if early > 0.5 and late < (1 - self._clear_ratio):
            return _result(MotionPattern.CLEARING, "Obstacle clearing from path")

        # ── APPROACHING: growing area + centred ───────────────────────────────
        if len(areas) >= 4 and center_ratio > self._approach_ratio:
            trend = _linear_trend(areas)
            if trend > 0.003:
                return _result(
                    MotionPattern.APPROACHING,
                    f"Approaching (area +{trend:.4f}/frame)",
                )

        # ── CROSSING: strong lateral movement, stable area ────────────────────
        if len(cx_vals) >= 4:
            cx_trend = _linear_trend(cx_vals)
            area_trend = _linear_trend(areas[-len(cx_vals) :])
            if abs(cx_trend) > 0.02 and abs(area_trend) < 0.003:
                direction = "left→right" if cx_trend > 0 else "right→left"
                return _result(
                    MotionPattern.CROSSING, f"Crossing ({direction})"
                )

        # ── BLOCKING: large, centred, stationary ─────────────────────────────
        recent_area = float(np.mean(areas[max(0, n - 5) :])) if areas else 0.0
        if present_ratio > 0.7 and center_ratio > 0.5 and recent_area > 0.05:
            return _result(
                MotionPattern.BLOCKING,
                f"Static obstacle blocking (area={recent_area:.2f})",
            )

        return _result(MotionPattern.UNCERTAIN, "Ambiguous pattern")


def detection_to_state(det_result, assumed_width: int = 400) -> FrameObstacleState:
    """Convert a DetectionResult into a FrameObstacleState for the recogniser."""
    if not det_result.boxes:
        return FrameObstacleState(False, False, 0.0, 0.5)

    # Use the largest obstacle box
    largest_idx = int(
        np.argmax([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in det_result.boxes])
    )
    x1, y1, x2, y2 = det_result.boxes[largest_idx]
    w = det_result.frame_width or assumed_width
    cx_norm = float(np.clip((x1 + x2) / 2 / w, 0.0, 1.0))

    return FrameObstacleState(
        obstacle_present=True,
        in_center=det_result.obstacle_in_center,
        area_frac=det_result.closest_area,
        centroid_x=cx_norm,
    )


def _result(pattern: MotionPattern, desc: str) -> TemporalResult:
    return TemporalResult(
        pattern=pattern,
        temporal_risk=PATTERN_RISK[pattern],
        description=desc,
    )


def _linear_trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    return float(np.polyfit(x, y, 1)[0])
