"""
speed_governor.py – kinematic safe-speed governor (proactive collision avoidance).

Caps the navigation action so the robot never travels faster than it can stop
within the confirmed-clear distance ahead, accounting for the pipeline's reaction
latency. This is the driver's-ed "total stopping distance" model — thinking
distance + braking distance:

    d_stop(v) = v * t_react  +  v^2 / (2 * a_max)  +  margin
                └ reaction ┘    └── braking ────┘    └ buffer ┘

Given the clear distance d (metres) ahead and the measured reaction time t_react
(seconds), it returns the FASTEST action whose stopping distance still fits:

    d >= d_stop(v_forward) → FORWARD is safe
    d >= d_stop(v_slow)    → SLOW
    else                   → STOP

Why this helps:
  • proactive     – speed is capped as a smooth function of distance, so the robot
                    slows *before* an obstacle is close, not at the last cm.
  • latency-aware – the `v * t_react` term reserves distance for the AI's decision
                    time, so a slow inference (e.g. V-JEPA 2 on CPU) forces a lower
                    safe speed instead of causing a crash.
  • reduce impact – set v_f > 0 (target_speed_mps) to brake to a slow crawl rather
                    than a dead stop where a graze is unavoidable.

Units are SI throughout (metres, seconds, m/s, m/s^2). forward_speed_mps and
slow_speed_mps are the robot's MEASURED travel speed at the FORWARD and SLOW
actions — they must be calibrated on the real robot (see README). The governor
only ever makes the action MORE cautious; it never speeds the robot up.
"""

from __future__ import annotations

from dataclasses import dataclass

from decision import Action   # reuse the same action enum (one-way import)


@dataclass
class GovernorDecision:
    action: Action              # fastest action that can still stop in time
    d_stop_forward_m: float     # stopping distance at the FORWARD speed
    d_stop_slow_m: float        # stopping distance at the SLOW speed
    reaction_s: float           # reaction time actually used (clamped)


class SpeedGovernor:
    def __init__(self, cfg: dict):
        g = (cfg.get("decision", {}) or {}).get("governor", {}) or {}
        self.enabled = bool(g.get("enabled", True))
        # Calibrated robot speeds (m/s) at the FORWARD and SLOW actions.
        self._v_fwd = float(g.get("forward_speed_mps", 0.35))
        self._v_slow = float(g.get("slow_speed_mps", 0.18))
        # Max achievable deceleration (m/s^2) and a fixed safety buffer (m).
        self._a = max(1e-3, float(g.get("max_decel_mps2", 0.6)))
        self._margin = float(g.get("safety_margin_m", 0.10))
        # Speed we aim to be at right before contact (0 = full stop). >0 "reduces
        # impact" instead of requiring a complete stop.
        self._v_target = float(g.get("target_speed_mps", 0.0))
        # Clamp the measured reaction time to a sane band.
        self._t_min = float(g.get("min_reaction_s", 0.2))
        self._t_max = float(g.get("max_reaction_s", 3.0))

    def stopping_distance_m(self, v_mps: float, t_react_s: float) -> float:
        """Total stopping distance from v to the target speed, incl. reaction time."""
        t = self._clamp_reaction(t_react_s)
        v = max(0.0, v_mps)
        vt = max(0.0, self._v_target)
        # Braking term uses (v^2 - v_target^2) so a nonzero target reduces the
        # required braking distance (we only need to slow to v_target, not to 0).
        braking = max(0.0, (v * v - vt * vt)) / (2.0 * self._a)
        return v * t + braking + self._margin

    def max_action(self, clear_distance_m: float, t_react_s: float) -> Action:
        """Fastest action whose stopping distance fits within clear_distance_m."""
        d = clear_distance_m
        if d >= self.stopping_distance_m(self._v_fwd, t_react_s):
            return Action.FORWARD
        if d >= self.stopping_distance_m(self._v_slow, t_react_s):
            return Action.SLOW
        return Action.STOP

    def evaluate(self, clear_distance_m: float, t_react_s: float) -> GovernorDecision:
        return GovernorDecision(
            action=self.max_action(clear_distance_m, t_react_s),
            d_stop_forward_m=self.stopping_distance_m(self._v_fwd, t_react_s),
            d_stop_slow_m=self.stopping_distance_m(self._v_slow, t_react_s),
            reaction_s=self._clamp_reaction(t_react_s),
        )

    def _clamp_reaction(self, t: float) -> float:
        return max(self._t_min, min(self._t_max, float(t)))
