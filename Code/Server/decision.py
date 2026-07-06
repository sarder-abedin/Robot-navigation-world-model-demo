"""
decision.py – Risk fusion and navigation action selection (server side).

Three signals → one fused risk score → one action:

  detector_risk    (instantaneous, from YOLO11 bounding boxes)
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
    BACKUP = "BACKUP"      # short reverse (obstacle too close / rushing in to turn)


# How cautious each action is (higher = more cautious). Used to cap a forward
# action to a more conservative one without ever speeding the robot up.
_CAUTION = {
    Action.FORWARD: 0, Action.SLOW: 1,
    Action.STOP: 2, Action.REROUTE: 2, Action.BACKUP: 2,
}


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

        # Closed-loop, context-aware reroute (wait / turn-until-clear / backup).
        rr = dec.get("reroute", {}) or {}
        self._rr_closed_loop = bool(rr.get("closed_loop", True))
        self._rr_wait_timeout = float(rr.get("wait_timeout_seconds", 2.0))
        self._rr_max_turn = float(rr.get("max_turn_seconds", 4.0))
        self._rr_backup_m = float(rr.get("backup_distance_m", 0.35))
        self._rr_backup_max = float(rr.get("max_backup_seconds", 1.0))
        # A side must beat the centre free-space to turn toward it — expressed both
        # as an absolute margin AND (for uncalibrated, relative-scale depth where
        # per-side gaps are only a few percent) a relative fraction of the centre
        # distance. Either one triggers a turn; otherwise straight ahead is the
        # clearest → STOP/search (don't turn into a wall). The absolute default is
        # small on purpose: a large one (the old 0.3 m) is unreachable for a depth
        # camera whose regions differ by centimetres, so the robot never turned.
        self._rr_dir_margin = float(rr.get("direction_margin_m", 0.05))
        self._rr_dir_frac = float(rr.get("direction_margin_frac", 0.10))
        self._rr_dynamic = set(rr.get("dynamic_classes", ["person", "cat", "dog"]))
        # How long to hold the ultrasonic reflex STOP before escalating to a
        # maneuver (turn/back-up) to go around an obstacle that won't clear.
        self._sonic_escalate_s = float(rr.get("ultrasonic_escalate_seconds", 1.5))
        # Once maneuvering around a sonar-blocking obstacle, stay committed until
        # the ultrasonic risk drops below this (front clear by a margin) before
        # resuming forward — hysteresis that stops the forward/backward oscillation
        # at the stop threshold. 1.0 = at the stop distance; lower = more clearance.
        self._sonic_resume_risk = float(rr.get("ultrasonic_resume_risk", 0.5))
        self._wait_since: float = 0.0   # when the current WAIT started (0 = not waiting)
        self._turn_since: float = 0.0   # when the current TURN started (0 = not turning)
        self._backup_since: float = 0.0 # when the current BACKUP run started
        self._blocked_since: float = 0.0  # when we first got boxed-in with no open side
        self._sonic_block_since: float = 0.0  # when the ultrasonic first hard-stopped us
        self._sonic_maneuvering: bool = False  # committed to clearing a sonar obstacle

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
        obstacle_label: str = "",
        depth_left_m: float | None = None,
        depth_center_m: float | None = None,
        depth_right_m: float | None = None,
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
            # Reflex STOP. But an obstacle the SONAR sees may never raise the
            # *vision* risk (YOLO can't classify a wall → det=0; motion STATIC_CLEAR
            # → ta=0), so the vision-driven reroute below would never fire and the
            # robot would sit here forever. So: hold the reflex STOP briefly, then —
            # if the obstacle won't clear — escalate to the closed-loop avoidance
            # (turn toward an open side / back up / rotate-to-search) to go around it.
            if self._sonic_block_since == 0.0:
                self._sonic_block_since = now
            if (self._rr_closed_loop
                    and now - self._sonic_block_since > self._sonic_escalate_s):
                self._sonic_maneuvering = True   # commit: see the hysteresis below
                action, reroute_dir, why = self._avoidance(
                    now, temporal_pattern, obstacle_label,
                    depth_left_m, depth_center_m, depth_right_m,
                )
                return self._result(
                    action, smoothed, detector_risk, world_model_risk,
                    temporal_risk, world_model_label, temporal_pattern,
                    f"Ultrasonic block won't clear → {why}",
                    reroute_direction=reroute_dir if action == Action.REROUTE else "",
                )
            self._stop_until = now + self._stop_hold
            # A hard-stop supersedes any in-progress avoidance maneuver; clear the
            # timers so that when risk resumes we start the maneuver fresh rather
            # than mid-way (e.g. skipping a WAIT because its timeout "already" elapsed
            # while the robot was actually stopped).
            self._wait_since = self._turn_since = self._backup_since = 0.0
            return self._result(
                Action.STOP, smoothed, detector_risk, world_model_risk,
                temporal_risk, world_model_label, temporal_pattern,
                "Ultrasonic hard-stop (obstacle within safe distance)",
            )

        # Sonar below the hard-stop threshold. If we were maneuvering around a
        # blocking obstacle, stay COMMITTED until the front is clear by a margin
        # (hysteresis): resume forward only once ultrasonic_risk falls below
        # ultrasonic_resume_risk. Otherwise a momentary clear — the back-up phase,
        # or the obstacle grazing the threshold — flips us straight to FORWARD and
        # we drive right back in: the forward/backward oscillation from the logs.
        if self._sonic_maneuvering:
            if ultrasonic_risk > self._sonic_resume_risk:
                action, reroute_dir, why = self._avoidance(
                    now, temporal_pattern, obstacle_label,
                    depth_left_m, depth_center_m, depth_right_m,
                )
                return self._result(
                    action, smoothed, detector_risk, world_model_risk,
                    temporal_risk, world_model_label, temporal_pattern,
                    f"clearing obstacle (committed until clear) → {why}",
                    reroute_direction=reroute_dir if action == Action.REROUTE else "",
                )
            self._sonic_maneuvering = False   # clear by a margin → resume normal nav
        # Sonar clear → reset the persistent-block timer so the next block starts fresh.
        self._sonic_block_since = 0.0

        # ── 2. Vision-driven action from the fused AI risk ────────────────────
        reroute_dir = ""
        if now < self._stop_until:
            action, explanation = Action.STOP, "Stop hold active"
            self._wait_since = self._turn_since = self._backup_since = 0.0  # stopped → reset avoidance
        elif smoothed <= self._low_max:
            action, explanation = Action.FORWARD, f"Low risk ({smoothed:.2f}) – forward"
            self._wait_since = self._turn_since = self._backup_since = self._blocked_since = 0.0  # clear
        elif smoothed <= self._med_max:
            action, explanation = Action.SLOW, f"Medium risk ({smoothed:.2f}) – slowing"
            self._wait_since = self._turn_since = self._backup_since = self._blocked_since = 0.0
        elif self._rr_closed_loop:
            # High vision risk → closed-loop, context-aware avoidance: wait out a
            # crossing obstacle, back off from one rushing in, or turn toward the
            # open side and keep turning until the gap opens.
            action, reroute_dir, explanation = self._avoidance(
                now, temporal_pattern, obstacle_label,
                depth_left_m, depth_center_m, depth_right_m,
            )
        else:
            # Legacy one-shot reroute: turn only when vision confirms a blocking
            # obstacle (V-JEPA 2 BLOCKED / temporal BLOCKING); else stop.
            if temporal_pattern == "BLOCKING" or world_model_label == "BLOCKED":
                action = Action.REROUTE
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

    # ── Closed-loop, context-aware avoidance ──────────────────────────────────

    def _turn_side(self, dl, dc, dr):
        """Pick a turn side from per-side depth. Returns (side, mode):
          • ("left"/"right", "turn") — that side is clearly more open than centre
          • ("", "turn") — geometry unknown (blind) → caller's default spin
          • ("", "stop") — straight ahead is the clearest → don't turn into a wall
        """
        if dl is None or dr is None or dc is None:
            return "", "turn"                       # unknown → legacy default turn
        best = max(dl, dr)
        gain = best - dc
        # A side is "clearly more open" if it beats centre by the absolute margin
        # OR by the relative fraction of the centre distance. The relative test is
        # what makes this work on uncalibrated depth (gaps of a few cm / percent).
        if gain > self._rr_dir_margin or (dc > 0 and gain > dc * self._rr_dir_frac):
            return ("left" if dl >= dr else "right"), "turn"
        return "", "stop"                           # centre is (roughly) the most open

    def _select_behaviour(self, pattern, obj_label, dc, dl, dr):
        """Choose an avoidance INTENT from motion + object + geometry.

        Returns (intent, direction) with intent in {WAIT, TURN, BACKUP, STOP_BLOCKED}:
          • too close + approaching → BACKUP (turning would steer into it)
          • crossing / clearing, or a dynamic obstacle (person) approaching but
            not close → WAIT (the path is likely to clear itself)
          • a side clearly more open than centre → TURN toward it
          • otherwise (straight ahead is the clearest, but blocked) → STOP_BLOCKED
        """
        close = dc is not None and 0 <= dc < self._rr_backup_m
        if pattern == "APPROACHING" and close:
            return "BACKUP", ""
        if pattern in ("CROSSING", "CLEARING"):
            return "WAIT", ""
        if pattern == "APPROACHING" and obj_label in self._rr_dynamic and not close:
            return "WAIT", ""   # e.g. a person ahead — pause, they may move aside
        side, mode = self._turn_side(dl, dc, dr)
        return ("TURN", side) if mode == "turn" else ("STOP_BLOCKED", "")

    def _avoidance(self, now, pattern, obj_label, dl, dc, dr):
        """Stateful closed-loop maneuver: WAIT (with timeout) / TURN-until-clear
        (with a spin guard) / BACKUP / STOP-when-blocked-with-no-clearer-side.
        Returns (action, reroute_dir, explanation)."""
        intent, direction = self._select_behaviour(pattern, obj_label, dc, dl, dr)

        if intent == "WAIT":
            self._blocked_since = 0.0
            if self._wait_since == 0.0:
                self._wait_since = now
            if now - self._wait_since <= self._rr_wait_timeout:
                self._turn_since = self._backup_since = 0.0
                return Action.STOP, "", f"wait ({pattern.lower()} — path may clear)"
            # waited long enough and still blocked → turn if a side is open, else stop
            side, mode = self._turn_side(dl, dc, dr)
            intent, direction = ("TURN", side) if mode == "turn" else ("STOP_BLOCKED", "")
        self._wait_since = 0.0

        if intent == "STOP_BLOCKED":
            # Boxed in: no side clearly more open. Don't freeze here forever (the old
            # behaviour — the robot just sat in STOP). Hold briefly so the sensors can
            # update, then rotate in place to SEARCH for an opening, capped by the
            # spin guard; if a full sweep finds nothing, re-hold and search again.
            self._backup_since = 0.0
            if self._blocked_since == 0.0:
                self._blocked_since = now
            if now - self._blocked_since <= self._stop_hold:
                self._turn_since = 0.0
                return Action.STOP, "", "blocked ahead, no clearer side — stop & reassess"
            if self._turn_since == 0.0:
                self._turn_since = now
            if now - self._turn_since > self._rr_max_turn:
                self._turn_since = 0.0
                self._blocked_since = 0.0
                self._stop_until = now + self._stop_hold
                return Action.STOP, "", "searched all around, still blocked — stop & reassess"
            side, _ = self._turn_side(dl, dc, dr)   # marginal side if any, else default spin
            return Action.REROUTE, side, "no clearly-open side — rotating to search for an opening"
        self._blocked_since = 0.0

        if intent == "BACKUP":
            self._turn_since = 0.0
            # The robot has no rear sensor, so don't reverse blindly forever: cap
            # the backup run, then STOP and reassess (front sensors take over).
            if self._backup_since == 0.0:
                self._backup_since = now
            if now - self._backup_since > self._rr_backup_max:
                self._backup_since = 0.0
                self._stop_until = now + self._stop_hold
                return Action.STOP, "", "backed up enough (no rear sensor) — stop & reassess"
            return Action.BACKUP, "", "backup (obstacle too close / approaching to turn)"
        self._backup_since = 0.0

        # TURN, closed-loop: keep turning toward the open side until the gap opens
        # (the FORWARD/SLOW branch resets _turn_since when it clears); a guard
        # stops an endless spin if no gap is ever found.
        if self._turn_since == 0.0:
            self._turn_since = now
        if now - self._turn_since > self._rr_max_turn:
            self._turn_since = 0.0
            self._stop_until = now + self._stop_hold
            return Action.STOP, "", "turned too long without a gap — stop & reassess"
        return Action.REROUTE, direction, f"turn {direction or 'default'} until gap opens"

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
