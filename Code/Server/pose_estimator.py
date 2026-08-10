"""
pose_estimator.py – open-loop dead-reckoning of the robot's world pose.

The robot has NO odometry (no wheel encoders, no IMU), so its position in the
world is not measured — it is *estimated* by integrating the commanded action
over time: action → calibrated speed × dt. FORWARD/SLOW advance along the
current heading, BACKUP reverses, TURN/REROUTE rotate in place. This is a
pure, framework-agnostic module (no torch / no PyQt) so it is unit-testable;
the AI pipeline calls update() once per processed frame and ships the resulting
pose to the UI, which anchors the trajectory map to it.

Because it is open-loop, the estimate DRIFTS: every un-modelled slip, ramp, or
turn-rate error accumulates without bound (there is no sensor to correct it).
The map is therefore "best-effort local", not a survey-grade world map — good
for showing where the robot has roughly been and where obstacles roughly are,
not for closing large loops.

Frame convention (matches Code/Client/nav_map.py's robot frame at heading 0):
  • the world origin (0, 0) is where the robot started,
  • +Y is the robot's initial forward ("up" on screen), +X is its initial right,
  • heading_deg is 0 when the robot faces +Y and INCREASES when it turns RIGHT
    (toward +X) — the same sign convention as goal_navigator's bearing_deg, so a
    positive bearing and a positive heading both mean "to the right".
The forward unit vector in the world is therefore (sin θ, cos θ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A single processed frame should represent a small slice of wall-clock time.
# Clamp dt so a stalled pipeline (a multi-second gap between frames — model
# warmup, a GC pause) doesn't teleport the robot metres across the map on the
# next tick. Anything longer is treated as this cap.
MAX_DT_S = 1.0


@dataclass
class Pose:
    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0

    def copy(self) -> "Pose":
        return Pose(self.x_m, self.y_m, self.heading_deg)


def _action_str(action) -> str:
    """Normalise an Action Enum / (str, Enum) member / plain string to UPPER."""
    val = getattr(action, "value", action)
    return str(val).strip().upper()


class PoseEstimator:
    """Integrate (x, y, heading) from commanded actions × calibrated speeds × dt."""

    def __init__(self, forward_speed_mps: float = 0.15, slow_speed_mps: float = 0.08,
                 backup_speed_mps: float = 0.10, turn_rate_dps: float = 45.0):
        self._v_fwd = max(0.0, float(forward_speed_mps))
        self._v_slow = max(0.0, float(slow_speed_mps))
        self._v_back = max(0.0, float(backup_speed_mps))
        self._turn_dps = max(0.0, float(turn_rate_dps))
        self._pose = Pose()

    @classmethod
    def from_config(cls, cfg: dict) -> "PoseEstimator":
        """Build from config.yaml — reuses the governor's calibrated forward/slow
        speeds (single source of truth) and the pose block's turn rate + backup."""
        gov = ((cfg or {}).get("decision", {}) or {}).get("governor", {}) or {}
        pose = (cfg or {}).get("pose", {}) or {}
        return cls(
            forward_speed_mps=gov.get("forward_speed_mps", 0.15),
            slow_speed_mps=gov.get("slow_speed_mps", 0.08),
            backup_speed_mps=pose.get("backup_speed_mps",
                                      gov.get("forward_speed_mps", 0.15)),
            turn_rate_dps=pose.get("turn_rate_dps", 45.0),
        )

    def reset(self) -> None:
        """Re-origin the pose (UI 'Reset map' — start a fresh trajectory)."""
        self._pose = Pose()

    @property
    def pose(self) -> Pose:
        return self._pose.copy()

    def update(self, action, direction: str = "", dt_s: float = 0.0) -> Pose:
        """Advance the pose by one action tick and return a copy of the new pose.

        action    : Action Enum / string (FORWARD, SLOW, STOP, BACKUP, TURN, REROUTE)
        direction : "left" | "right" | "" — turn side for TURN / REROUTE
        dt_s      : wall-clock seconds since the previous update (clamped to MAX_DT_S)
        """
        try:
            dt = float(dt_s)
        except (TypeError, ValueError):
            dt = 0.0
        if dt <= 0.0:
            return self._pose.copy()
        dt = min(dt, MAX_DT_S)

        act = _action_str(action)
        th = math.radians(self._pose.heading_deg)

        if act in ("FORWARD", "SLOW"):
            v = self._v_fwd if act == "FORWARD" else self._v_slow
            s = v * dt
            self._pose.x_m += s * math.sin(th)
            self._pose.y_m += s * math.cos(th)
        elif act == "BACKUP":
            s = self._v_back * dt
            self._pose.x_m -= s * math.sin(th)
            self._pose.y_m -= s * math.cos(th)
        elif act in ("TURN", "REROUTE"):
            # In-place rotation toward the open side. REROUTE is a back-up-then-spin
            # on the robot; the spin dominates the net heading change, so we model
            # it as a pure rotation here (the brief reverse is left un-modelled — a
            # small, bounded under-estimate of travel, not a drift multiplier).
            sign = 1.0 if (direction or "").strip().lower() == "right" else -1.0
            self._pose.heading_deg = _wrap_deg(
                self._pose.heading_deg + sign * self._turn_dps * dt)
        # STOP (and anything unrecognised) → no motion.

        return self._pose.copy()


def _wrap_deg(deg: float) -> float:
    """Wrap an angle to (-180, 180] so the heading never grows without bound."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d
