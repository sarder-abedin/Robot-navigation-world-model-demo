"""
nav_map.py – geometry + status parsing for the 2D egocentric navigation map.

This is the framework-agnostic core of the map the PyQt viewer draws: it turns a
CMD_AISTATUS line into a MapModel and converts robot-relative polar readings
(distance + bearing) into world/screen coordinates. It imports no PyQt, so the
logic is unit-testable without a display (the QPainter widget in ai_viewer.py is
the thin drawing layer on top).

Frame convention (robot-centric / egocentric — there is no odometry, so the map
is always "right now", never world-anchored):
  • the robot sits at the origin, facing +Y ("up" on screen),
  • +X is to the robot's right, bearing is degrees from straight-ahead
    (negative = left, positive = right), matching goal_navigator's bearing_deg
    and the LEFT/CENTER/RIGHT depth thirds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# CMD_AISTATUS field layout (see ai_pipeline._broadcast_status). Trailing fields
# were appended over time; older servers omit them, so parsing tolerates a short
# line. '#' never appears inside a field (ssv2 is sanitised server-side).
#   0 CMD_AISTATUS  1 action     2 risk_pct   3 wm_label    4 pattern
#   5 sonic_cm      6 ssv2       7 clear_dist 8 clear_dir   9 goal_status
#  10 depth_left_m 11 depth_right_m 12 goal_bearing_deg 13 goal_dist_m


@dataclass
class MapModel:
    action: str = "---"
    risk_pct: int = 0
    sonic_m: float | None = None        # None = no echo / unknown (never drawn at 0)
    depth_left_m: float | None = None
    depth_center_m: float | None = None
    depth_right_m: float | None = None
    clear_dir: str = ""                 # LEFT | CENTER | RIGHT | ""
    goal_status: str = "none"           # none | tracking | lost | reached
    goal_bearing_deg: float = 0.0
    goal_dist_m: float | None = None
    max_range_m: float = 5.0
    fov_deg: float = 66.0

    @property
    def has_goal(self) -> bool:
        return self.goal_status not in ("", "none")


def _pos_float(s: str):
    """Parse a float; return None for blanks, non-numbers, non-finite (nan/inf),
    or ≤0 sentinels (-1). Rejecting inf/nan matters: they reach the QPainter
    widget as arc angles (int(round(inf)) → OverflowError, round(nan) → ValueError)."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if (math.isfinite(v) and v > 0) else None


def parse_status(line: str, max_range_m: float = 5.0, fov_deg: float = 66.0) -> MapModel:
    """Parse a CMD_AISTATUS line into a MapModel (tolerant of older short lines)."""
    parts = line.strip().split("#")
    m = MapModel(max_range_m=max_range_m, fov_deg=fov_deg)
    if len(parts) < 6 or parts[0] != "CMD_AISTATUS":
        return m
    m.action = parts[1] or "---"
    try:
        m.risk_pct = int(parts[2])
    except (TypeError, ValueError):
        m.risk_pct = 0
    son = _pos_float(parts[5])           # cm on the wire
    m.sonic_m = (son / 100.0) if son is not None else None
    if len(parts) > 7:
        m.depth_center_m = _pos_float(parts[7])
    if len(parts) > 8:
        m.clear_dir = (parts[8] or "").strip().upper()
    if len(parts) > 9:
        m.goal_status = (parts[9] or "none").strip() or "none"
    if len(parts) > 10:
        m.depth_left_m = _pos_float(parts[10])
    if len(parts) > 11:
        m.depth_right_m = _pos_float(parts[11])
    if len(parts) > 12:
        try:
            gb = float(parts[12])
            m.goal_bearing_deg = gb if math.isfinite(gb) else 0.0
        except (TypeError, ValueError):
            m.goal_bearing_deg = 0.0
    if len(parts) > 13:
        m.goal_dist_m = _pos_float(parts[13])
    return m


def polar_to_world(dist_m: float, bearing_deg: float) -> tuple[float, float]:
    """(distance, bearing°) → (x_right_m, y_forward_m) in the robot frame."""
    th = math.radians(bearing_deg)
    return dist_m * math.sin(th), dist_m * math.cos(th)


def world_to_screen(x_m: float, y_m: float, origin_px: tuple[float, float],
                    px_per_m: float) -> tuple[float, float]:
    """Robot-frame metres → screen pixels (origin at the robot, +Y up on screen)."""
    ox, oy = origin_px
    return ox + x_m * px_per_m, oy - y_m * px_per_m


def fit_scale(width_px: float, height_px: float, margin_px: float,
              max_range_m: float, fov_deg: float) -> float:
    """Pixels-per-metre so the FOV fan at max range fits both axes with a margin."""
    max_range_m = max(0.1, max_range_m)
    v = (height_px - 2 * margin_px) / max_range_m            # forward extent
    half_w = max(1e-3, math.sin(math.radians(fov_deg / 2.0)) * max_range_m)
    h = (width_px / 2.0 - margin_px) / half_w                # widest lateral extent
    return max(1.0, min(v, h))


# Bearings (deg) of the LEFT/CENTER/RIGHT depth thirds within the camera FOV.
def sector_center_bearings(fov_deg: float) -> dict:
    third = fov_deg / 3.0
    return {"LEFT": -third, "CENTER": 0.0, "RIGHT": third}


def sector_edge_bearings(fov_deg: float, sector: str) -> tuple[float, float]:
    """(start°, end°) span of a LEFT/CENTER/RIGHT third across the FOV."""
    h = fov_deg / 2.0
    thirds = {"LEFT": (-h, -h / 3.0), "CENTER": (-h / 3.0, h / 3.0), "RIGHT": (h / 3.0, h)}
    return thirds.get(sector.upper(), (-h, h))


def proximity_color(dist_m, near: float = 0.4, far: float = 2.5) -> tuple[int, int, int]:
    """RGB for a distance: red when near, green when far, grey when unknown."""
    if dist_m is None or dist_m < 0:
        return (120, 120, 120)
    t = max(0.0, min(1.0, (dist_m - near) / (far - near)))
    r = int(round(220 * (1 - t) + 40 * t))
    g = int(round(40 * (1 - t) + 185 * t))
    return (r, g, 45)
