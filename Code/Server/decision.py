"""
decision.py – Risk fusion and navigation action selection (server side).

Three signals → one fused risk score → one action:

  detector_risk    (instantaneous, from YOLOv8 bounding boxes)
  world_model_risk (predictive,    from V-JEPA 2 future-embedding comparison)
  temporal_risk    (trajectory,    from SSv2-style motion-pattern rules)

The fused score passes through a hysteresis filter so the robot does not
oscillate near threshold boundaries.

Baseline vs Predictive
──────────────────────
baseline   → world_model weight = 0; temporal weight halved.
             Robot reacts only to what is currently in frame.
predictive → all three weights active; V-JEPA 2 can trigger SLOW even when
             the detector still reports low risk (early-warning deceleration).

This difference is the core of the demo: predictive mode visibly starts
braking earlier and makes smoother transitions.
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


# How cautious each action is (higher = more cautious). Used to cap a forward
# action to a more conservative one without ever speeding the robot up.
_CAUTION = {Action.FORWARD: 0, Action.SLOW: 1, Action.STOP: 2, Action.REROUTE: 2}


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
    reroute_direction: str = ""   # "left"/"right" for REROUTE (from depth), else ""


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

        # Kinematic safe-speed governor (proactive, latency-aware). Lazy import
        # keeps decision.py free of a top-level dependency cycle (speed_governor
        # imports Action from here).
        from speed_governor import SpeedGovernor
        self._governor = SpeedGovernor(cfg)

        if navigation_mode == "baseline":
            # Zero out the predictive signals so comparison is fair
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
        clear_distance_m: float | None = None,
        reaction_s: float = 0.0,
        clear_direction: str | None = None,
    ) -> DecisionResult:
        # ── AI risk fusion (vision only) ──────────────────────────────────────
        # The ultrasonic is NOT mixed in here — it is a separate deterministic
        # safety layer applied below. This keeps the probabilistic AI risk
        # (which drives FORWARD/SLOW/REROUTE) independent of the hard-stop reflex.
        fused = (
            self._w_det * detector_risk
            + self._w_wm * world_model_risk
            + self._w_ta * temporal_risk
        )
        fused = float(min(max(fused, 0.0), 1.0))

        # Hysteresis: allow risk to climb immediately, require margin to drop
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

        # ── 1. Ultrasonic hard-stop (deterministic safety override) ───────────
        # ultrasonic_risk reaches 1.0 only when the sensor reports an obstacle
        # within the stop distance (or is blind-close). This is a reflex, decided
        # by distance alone — separate from and higher priority than the AI risk.
        if ultrasonic_risk >= 1.0:
            self._stop_until = now + self._stop_hold
            return self._result(
                Action.STOP, smoothed, detector_risk, world_model_risk,
                temporal_risk, world_model_label, temporal_pattern,
                "Ultrasonic hard-stop (obstacle within safe distance)",
            )

        # ── 2. Vision-driven action from the fused AI risk ────────────────────
        reroute_dir = ""
        if now < self._stop_until:
            action, explanation = Action.STOP, "Stop hold active"
        elif smoothed <= self._low_max:
            action, explanation = Action.FORWARD, f"Low risk ({smoothed:.2f}) – forward"
        elif smoothed <= self._med_max:
            action, explanation = Action.SLOW, f"Medium risk ({smoothed:.2f}) – slowing"
        else:
            # High vision risk → the CAMERA decides stop vs turn. Reroute only
            # when vision indicates a blocking obstacle (V-JEPA 2 BLOCKED or a
            # temporal BLOCKING pattern), since the ultrasonic can't say which
            # way is clear.
            if temporal_pattern == "BLOCKING" or world_model_label == "BLOCKED":
                action = Action.REROUTE
                # Depth's clear direction (LEFT/RIGHT) tells the robot WHICH way
                # to turn; CENTER/None → "" so the Pi uses its default spin.
                reroute_dir = {"LEFT": "left", "RIGHT": "right"}.get(
                    (clear_direction or "").upper(), "")
                explanation = (
                    f"High risk ({smoothed:.2f}) – vision reroute "
                    f"(wm={world_model_label}, pattern={temporal_pattern}, "
                    f"turn={reroute_dir or 'default'})"
                )
            else:
                action = Action.STOP
                self._stop_until = now + self._stop_hold
                explanation = f"High risk ({smoothed:.2f}) + {temporal_pattern} – stop"

        # V-JEPA 2 early-warning: world model predicts BLOCKED but detector
        # hasn't seen it yet → proactively decelerate from FORWARD to SLOW
        if (
            self._mode == "predictive"
            and world_model_label == "BLOCKED"
            and action == Action.FORWARD
        ):
            action = Action.SLOW
            explanation += " [WM early-warning]"

        # ── 3. Kinematic safe-speed governor (proactive, latency-aware) ───────
        # Only ever downgrades a forward-motion action so the robot can always
        # stop within the confirmed-clear distance given the AI's reaction time.
        # Needs a valid distance (metres); when the sensor is blind it's skipped
        # and the ultrasonic hard-stop / blind-hold above still apply.
        if (
            self._governor.enabled
            and clear_distance_m is not None
            and clear_distance_m >= 0
            and action in (Action.FORWARD, Action.SLOW)
        ):
            gov = self._governor.max_action(clear_distance_m, reaction_s)
            if _CAUTION[gov] > _CAUTION[action]:
                action = gov
                explanation += (
                    f" [governor→{gov.value}: d={clear_distance_m:.2f}m "
                    f"t_react={reaction_s:.2f}s]"
                )

        return self._result(
            action, smoothed, detector_risk, world_model_risk, temporal_risk,
            world_model_label, temporal_pattern, explanation,
            reroute_direction=reroute_dir if action == Action.REROUTE else "",
        )

    @staticmethod
    def _result(action, risk, det, wm, ta, wm_label, pattern, explanation,
                reroute_direction=""):
        return DecisionResult(
            action=action,
            risk_score=risk,
            detector_risk=det,
            world_model_risk=wm,
            temporal_risk=ta,
            world_model_label=wm_label,
            temporal_pattern=pattern,
            explanation=explanation,
            reroute_direction=reroute_direction,
        )
