"""
decision.py – Risk fusion and navigation action selection.

Three signals are combined into a single risk score:
  1. detector_risk   – instantaneous, from YOLOv8 bounding boxes
  2. world_model_risk – predictive, from V-JEPA 2 future-latent comparison
  3. temporal_risk   – motion-pattern, from SSv2-style trajectory analysis

The weighted sum is passed through a hysteresis filter so the robot does
not oscillate between actions when the score sits near a threshold.

Actions
───────
  FORWARD  – drive at full speed
  SLOW     – drive at reduced speed (medium risk zone)
  STOP     – halt and hold
  REROUTE  – halt, turn, then resume

Baseline vs Predictive mode
────────────────────────────
In *baseline* mode the world_model weight is zeroed and the temporal
weight is halved, so the robot reacts only to what is currently visible.
This lets a side-by-side comparison show that predictive mode anticipates
blockages earlier and produces smoother, safer trajectories.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Action(str, Enum):
    FORWARD = "FORWARD"
    SLOW = "SLOW"
    STOP = "STOP"
    REROUTE = "REROUTE"


@dataclass
class DecisionResult:
    action: Action
    risk_score: float            # fused risk in [0, 1]
    detector_risk: float
    world_model_risk: float
    temporal_risk: float
    world_model_label: str       # V-JEPA 2 prediction label
    temporal_pattern: str        # SSv2-style motion label
    explanation: str


class DecisionFuser:
    """
    Fuses the three risk signals and applies hysteresis to produce a stable
    navigation action.
    """

    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        dec_cfg = cfg["decision"]
        w = dec_cfg["weights"]
        self._w_det = w["detector"]
        self._w_wm = w["world_model"]
        self._w_ta = w["temporal"]
        self._low_max = dec_cfg["low_risk_max"]
        self._med_max = dec_cfg["medium_risk_max"]
        self._hysteresis = dec_cfg["hysteresis"]
        self._stop_hold = dec_cfg["stop_hold_seconds"]

        self._navigation_mode = navigation_mode
        self._last_action = Action.FORWARD
        self._last_risk = 0.0
        self._stop_until: float = 0.0   # monotonic time when stop-hold expires

        if navigation_mode == "baseline":
            # Reactive-only: ignore world model, halve temporal signal
            self._w_wm = 0.0
            self._w_ta /= 2
            total = self._w_det + self._w_ta
            self._w_det /= total
            self._w_ta /= total
            logger.info("DecisionFuser: BASELINE mode (world model disabled)")
        else:
            logger.info("DecisionFuser: PREDICTIVE mode")

    def decide(
        self,
        detector_risk: float,
        world_model_risk: float,
        temporal_risk: float,
        world_model_label: str = "UNKNOWN",
        temporal_pattern: str = "UNKNOWN",
    ) -> DecisionResult:
        # ── Weighted fusion ───────────────────────────────────────────────────
        fused = (
            self._w_det * detector_risk
            + self._w_wm * world_model_risk
            + self._w_ta * temporal_risk
        )
        fused = float(min(max(fused, 0.0), 1.0))

        # ── Hysteresis filter ─────────────────────────────────────────────────
        # Only allow upward jumps immediately; require crossing a margin before
        # moving back down.  This avoids rapid oscillation near boundaries.
        if fused > self._last_risk:
            smoothed = fused
        else:
            smoothed = (
                fused
                if (self._last_risk - fused) > self._hysteresis
                else self._last_risk
            )
        self._last_risk = smoothed

        # ── Action selection ──────────────────────────────────────────────────
        now = time.monotonic()
        if now < self._stop_until:
            # Held in stop phase
            action = Action.STOP
            explanation = "Stop hold active"
        elif smoothed <= self._low_max:
            action = Action.FORWARD
            explanation = f"Low risk ({smoothed:.2f}) – drive forward"
        elif smoothed <= self._med_max:
            action = Action.SLOW
            explanation = f"Medium risk ({smoothed:.2f}) – slowing down"
        else:
            # High risk: decide STOP vs REROUTE based on pattern
            if temporal_pattern in ("BLOCKING",):
                action = Action.REROUTE
                explanation = f"High risk ({smoothed:.2f}) + {temporal_pattern} – rerouting"
            else:
                action = Action.STOP
                self._stop_until = now + self._stop_hold
                explanation = f"High risk ({smoothed:.2f}) + {temporal_pattern} – stopping"

        # V-JEPA 2 prediction upgrading: if the world model already sees a
        # BLOCKED future but detector risk is still low (early warning), elevate
        # FORWARD → SLOW so the robot starts decelerating proactively.
        if (
            self._navigation_mode == "predictive"
            and world_model_label == "BLOCKED"
            and action == Action.FORWARD
        ):
            action = Action.SLOW
            explanation += " [WM early-warning: decelerating]"

        self._last_action = action
        return DecisionResult(
            action=action,
            risk_score=smoothed,
            detector_risk=detector_risk,
            world_model_risk=world_model_risk,
            temporal_risk=temporal_risk,
            world_model_label=world_model_label,
            temporal_pattern=temporal_pattern,
            explanation=explanation,
        )
