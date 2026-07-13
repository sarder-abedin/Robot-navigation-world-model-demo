"""
test_nav_map_rpi.py – geometry + CMD_AISTATUS parsing for the 2D navigation map.

Covers nav_map (the framework-agnostic core of the ai_viewer map): status
parsing (full + older short lines), polar→world→screen geometry, FOV sector
bearings and the proximity colour ramp. No PyQt/display needed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Client"))

import nav_map as nm


# ── parse_status ──────────────────────────────────────────────────────────────

FULL = ("CMD_AISTATUS#FORWARD#42#CLEAR#APPROACHING#123.0#person moving closer"
        "#1.80#LEFT#tracking#2.10#0.90#-15.0#3.40")


def test_parse_full_line():
    m = nm.parse_status(FULL)
    assert m.action == "FORWARD"
    assert m.risk_pct == 42
    assert math.isclose(m.sonic_m, 1.23)          # cm → m
    assert math.isclose(m.depth_center_m, 1.80)
    assert m.clear_dir == "LEFT"
    assert m.goal_status == "tracking"
    assert math.isclose(m.depth_left_m, 2.10)
    assert math.isclose(m.depth_right_m, 0.90)
    assert math.isclose(m.goal_bearing_deg, -15.0)
    assert math.isclose(m.goal_dist_m, 3.40)
    assert m.has_goal


def test_parse_old_short_line_defaults():
    # Pre-map server: only the first 9 fields, no depth-sides / goal geometry.
    old = "CMD_AISTATUS#STOP#88#BLOCKED#STATIC#40.0#obstacle#0.35#CENTER"
    m = nm.parse_status(old)
    assert m.action == "STOP" and m.depth_center_m == 0.35 and m.clear_dir == "CENTER"
    assert m.depth_left_m is None and m.depth_right_m is None   # absent → unknown
    assert m.goal_status == "none" and not m.has_goal
    assert m.goal_dist_m is None


def test_parse_sentinels_become_unknown():
    # -1 sentinels (no echo / depth off / no goal) must not render as 0-metre walls.
    line = "CMD_AISTATUS#SLOW#10#MIXED#CLEARING#-1.0#x#-1.00#none#-1.00#-1.00#0.0#-1.00"
    m = nm.parse_status(line)
    assert m.sonic_m is None
    assert m.depth_center_m is None and m.depth_left_m is None and m.depth_right_m is None
    assert m.goal_dist_m is None


def test_parse_garbage_is_safe():
    m = nm.parse_status("not a status line")
    assert m.action == "---" and m.risk_pct == 0 and m.sonic_m is None


# ── geometry ──────────────────────────────────────────────────────────────────

def test_polar_dead_ahead_is_pure_forward():
    x, y = nm.polar_to_world(2.0, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 2.0)


def test_polar_sign_left_right():
    xr, _ = nm.polar_to_world(1.0, +30.0)      # right → +x
    xl, _ = nm.polar_to_world(1.0, -30.0)      # left  → -x
    assert xr > 0 and xl < 0 and math.isclose(xr, -xl)


def test_world_to_screen_forward_is_up():
    # +Y (forward) must move UP on screen (smaller pixel y).
    origin = (100.0, 300.0)
    _sx, sy = nm.world_to_screen(0.0, 1.0, origin, 50.0)
    assert sy < origin[1]
    sx, _ = nm.world_to_screen(1.0, 0.0, origin, 50.0)   # +x right
    assert sx > origin[0]


def test_sector_bearings_partition_the_fov():
    assert nm.sector_center_bearings(66.0)["CENTER"] == 0.0
    assert nm.sector_center_bearings(66.0)["LEFT"] < 0 < nm.sector_center_bearings(66.0)["RIGHT"]
    # thirds tile [-33, +33] contiguously with no gap/overlap
    l = nm.sector_edge_bearings(66.0, "LEFT")
    c = nm.sector_edge_bearings(66.0, "CENTER")
    r = nm.sector_edge_bearings(66.0, "RIGHT")
    assert math.isclose(l[0], -33.0) and math.isclose(r[1], 33.0)
    assert math.isclose(l[1], c[0]) and math.isclose(c[1], r[0])


def test_fit_scale_keeps_fan_inside():
    ppm = nm.fit_scale(300, 315, 26, 5.0, 66.0)
    # widest point of the fan at max range must fit within half width - margin
    half_w = math.sin(math.radians(33.0)) * 5.0 * ppm
    assert half_w <= (300 / 2.0 - 26) + 1e-6
    # forward extent must fit the height
    assert 5.0 * ppm <= (315 - 2 * 26) + 1e-6


# ── colour ramp ───────────────────────────────────────────────────────────────

def test_proximity_colour_near_red_far_green_unknown_grey():
    near = nm.proximity_color(0.2)
    far = nm.proximity_color(3.0)
    unknown = nm.proximity_color(None)
    assert near[0] > near[1]        # red dominates when close
    assert far[1] > far[0]          # green dominates when far
    assert unknown == (120, 120, 120)
    assert nm.proximity_color(-1.0) == (120, 120, 120)   # sentinel → grey
