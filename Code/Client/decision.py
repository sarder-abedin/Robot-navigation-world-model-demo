"""
decision.py – Risk fusion and navigation action selection (client side).

Runs on the operator PC alongside V-JEPA 2 and the temporal recognizer.
Identical logic to Code/Server/decision.py — split here so the client
can import it without touching the Pi server's Python path.

See Code/Server/decision.py for full documentation.
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
    risk_score: float
    detector_risk: float
    world_model_risk: float
    temporal_risk: float
    world_model_label: str
    temporal_pattern: str
    explanation: str


class DecisionFuser:
    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        dec = cfg["decision"]
        w = dec["weights"]
        self._w_det = w["detector"]
        self._w_wm = w["world_model"]
        self._w_ta = w["temporal"]
        self._low_max = dec["low_risk_max"]
        self._med_max = dec["medium_risk_max"]
        self._hysteresis = dec["hysteresis"]
        self._stop_hold = dec["stop_hold_seconds"]

        self._mode = navigation_mode
        self._last_risk = 0.0
        self._stop_until: float = 0.0

        if navigation_mode == "baseline":
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
        ultrasonic_risk: float = 0.0,
    ) -> DecisionResult:
        fused = (
            self._w_det * detector_risk
            + self._w_wm * world_model_risk
            + self._w_ta * temporal_risk
        )
        fused = max(fused, ultrasonic_risk)
        fused = float(min(max(fused, 0.0), 1.0))

        if fused > self._last_risk:
            smoothed = fused
        else:
            smoothed = (
                fused
                if (self._last_risk - fused) > self._hysteresis
                else self._last_risk
            )
        self._last_risk = smoothed

        now = time.monotonic()
        if now < self._stop_until:
            action = Action.STOP
            explanation = "Stop hold active"
        elif smoothed <= self._low_max:
            action = Action.FORWARD
            explanation = f"Low risk ({smoothed:.2f}) – forward"
        elif smoothed <= self._med_max:
            action = Action.SLOW
            explanation = f"Medium risk ({smoothed:.2f}) – slowing"
        else:
            if temporal_pattern == "BLOCKING":
                action = Action.REROUTE
                explanation = f"High risk ({smoothed:.2f}) + BLOCKING – reroute"
            else:
                action = Action.STOP
                self._stop_until = now + self._stop_hold
                explanation = f"High risk ({smoothed:.2f}) + {temporal_pattern} – stop"

        if (
            self._mode == "predictive"
            and world_model_label == "BLOCKED"
            and action == Action.FORWARD
        ):
            action = Action.SLOW
            explanation += " [WM early-warning]"

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
