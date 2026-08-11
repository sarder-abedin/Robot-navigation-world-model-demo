"""
test_world_map_rpi.py – world-anchored map geometry, parsing and accumulation.

Covers world_map (framework-agnostic core of ai_viewer.WorldMapWidget): pose +
CMD_MAPOBJ parsing, polar→robot→world projection, trajectory/obstacle
accumulation with decimation + caps, moving-object flagging, goal placement,
view bounds and the label colour hash. No PyQt/display needed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Client"))

import world_map as wm


# A full CMD_AISTATUS line incl. the trailing pose fields (x, y, heading).
def _status(sonic_cm=-1.0, goal_status="none", goal_bearing=0.0, goal_dist=-1.0,
            px=0.0, py=0.0, pth=0.0, wm_label="CLEAR", clear_dist=-1.0):
    return (f"CMD_AISTATUS#FORWARD#10#{wm_label}#STATIC_CLEAR#{sonic_cm:.1f}#"
            f"#{clear_dist:.2f}#CENTER#{goal_status}#-1.00#-1.00#{goal_bearing:.1f}#{goal_dist:.2f}"
            f"#{px:.3f}#{py:.3f}#{pth:.1f}")


# ── parse_pose ──────────────────────────────────────────────────────────────

def test_parse_pose_full():
    p = wm.parse_pose(_status(px=1.5, py=-2.0, pth=30.0))
    assert math.isclose(p.x_m, 1.5) and math.isclose(p.y_m, -2.0)
    assert math.isclose(p.heading_deg, 30.0)


def test_parse_pose_absent_on_old_line():
    old = "CMD_AISTATUS#FORWARD#10#CLEAR#STATIC_CLEAR#50.0#"   # no pose fields
    assert wm.parse_pose(old) is None


def test_parse_pose_rejects_non_status():
    assert wm.parse_pose("CMD_MAPOBJ#chair,0,1") is None


# ── parse_mapobj ────────────────────────────────────────────────────────────

def test_parse_mapobj_multiple():
    objs = wm.parse_mapobj("CMD_MAPOBJ#person,-10.0,2.0;chair,15.0,-1.00")
    assert objs == [("person", -10.0, 2.0), ("chair", 15.0, None)]


def test_parse_mapobj_empty():
    assert wm.parse_mapobj("CMD_MAPOBJ#") == []


def test_parse_mapobj_skips_malformed():
    objs = wm.parse_mapobj("CMD_MAPOBJ#ok,1.0,2.0;garbage;bad,x,y;;good,3.0,4.0")
    assert objs == [("ok", 1.0, 2.0), ("good", 3.0, 4.0)]


def test_parse_mapobj_rejects_non_mapobj():
    assert wm.parse_mapobj("CMD_AISTATUS#...") == []


# ── geometry ────────────────────────────────────────────────────────────────

def test_polar_to_robot_straight_ahead():
    x, y = wm.polar_to_robot(2.0, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 2.0)


def test_polar_to_robot_right_is_positive_x():
    x, y = wm.polar_to_robot(1.0, 90.0)
    assert math.isclose(x, 1.0) and math.isclose(y, 0.0, abs_tol=1e-9)


def test_robot_to_world_identity_at_origin():
    pose = wm.WorldPose(0.0, 0.0, 0.0)
    assert wm.robot_to_world(1.0, 2.0, pose) == (1.0, 2.0)


def test_robot_to_world_translation():
    pose = wm.WorldPose(5.0, 5.0, 0.0)
    x, y = wm.robot_to_world(0.0, 2.0, pose)      # 2 m ahead of a robot at (5,5)
    assert math.isclose(x, 5.0) and math.isclose(y, 7.0)


def test_robot_to_world_rotated_90_right():
    """Facing +X (heading 90°): a point 2 m 'ahead' is +2 m in world X."""
    pose = wm.WorldPose(0.0, 0.0, 90.0)
    x, y = wm.robot_to_world(0.0, 2.0, pose)
    assert math.isclose(x, 2.0, abs_tol=1e-9) and math.isclose(y, 0.0, abs_tol=1e-9)


# ── accumulation ────────────────────────────────────────────────────────────

def test_update_status_builds_trajectory():
    m = wm.WorldModel()
    m.update_status(_status(px=0.0, py=0.0))
    m.update_status(_status(px=0.0, py=1.0))
    assert len(m.trajectory) == 2
    assert m.pose is not None and math.isclose(m.pose.y_m, 1.0)


def test_trajectory_decimates_tiny_moves():
    """Points closer than trail_step_m collapse (no thousands of stationary points)."""
    m = wm.WorldModel(trail_step_m=0.05)
    m.update_status(_status(px=0.0, py=0.0))
    m.update_status(_status(px=0.0, py=0.01))     # 1 cm move → below step
    m.update_status(_status(px=0.0, py=0.02))
    assert len(m.trajectory) == 1


def test_trajectory_cap_drops_oldest():
    m = wm.WorldModel(max_trail=3, trail_step_m=0.0)
    for i in range(6):
        m.update_status(_status(px=float(i), py=0.0))
    assert len(m.trajectory) == 3
    assert math.isclose(m.trajectory[-1].x_m, 5.0)   # newest kept
    assert math.isclose(m.trajectory[0].x_m, 3.0)    # oldest dropped


def test_ultrasonic_becomes_world_obstacle_ahead():
    m = wm.WorldModel()
    m.update_status(_status(sonic_cm=100.0, px=0.0, py=0.0, pth=0.0))
    assert len(m.obstacle_points) == 1
    x, y = m.obstacle_points[0]
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 1.0, abs_tol=1e-6)


def test_ultrasonic_none_when_no_echo():
    m = wm.WorldModel()
    m.update_status(_status(sonic_cm=-1.0))
    assert m.obstacle_points == []


def test_ultrasonic_merges_nearby_hits():
    """A stationary wall pinged repeatedly must not pile up duplicate points."""
    m = wm.WorldModel(obstacle_merge_m=0.1)
    for _ in range(5):
        m.update_status(_status(sonic_cm=100.0, px=0.0, py=0.0))
    assert len(m.obstacle_points) == 1


def test_goal_placed_in_world():
    m = wm.WorldModel()
    m.update_status(_status(goal_status="tracking", goal_bearing=0.0,
                            goal_dist=2.0, px=0.0, py=0.0))
    assert m.goal is not None
    assert math.isclose(m.goal[1], 2.0, abs_tol=1e-6)


def test_goal_cleared_when_none():
    m = wm.WorldModel()
    m.update_status(_status(goal_status="tracking", goal_dist=2.0))
    m.update_status(_status(goal_status="none"))
    assert m.goal is None


def test_update_objects_projects_to_world():
    m = wm.WorldModel()
    m.update_status(_status(px=0.0, py=0.0, pth=0.0))
    m.update_objects("CMD_MAPOBJ#chair,0.0,2.0")
    assert len(m.objects) == 1
    o = m.objects[0]
    assert o.label == "chair" and o.dist_known
    assert math.isclose(o.y_m, 2.0, abs_tol=1e-6)


def test_update_objects_unknown_distance_uses_fallback():
    m = wm.WorldModel()
    m.update_status(_status(px=0.0, py=0.0))
    m.update_objects("CMD_MAPOBJ#person,0.0,-1.00", fallback_dist_m=1.5)
    o = m.objects[0]
    assert not o.dist_known
    assert math.isclose(o.y_m, 1.5, abs_tol=1e-6)


def test_update_objects_ignored_before_pose():
    m = wm.WorldModel()
    m.update_objects("CMD_MAPOBJ#chair,0.0,2.0")      # no pose yet
    assert m.objects == []


def test_moving_flag_set_on_world_shift():
    """Same-label object that jumps in the world between frames → moving."""
    m = wm.WorldModel(move_thresh_m=0.2)
    m.update_status(_status(px=0.0, py=0.0, pth=0.0))
    m.update_objects("CMD_MAPOBJ#person,0.0,2.0")     # at (0,2)
    m.update_objects("CMD_MAPOBJ#person,0.0,3.0")     # jumped to (0,3): +1 m
    assert m.objects[0].moving


def test_static_object_not_flagged_moving():
    m = wm.WorldModel(move_thresh_m=0.2)
    m.update_status(_status(px=0.0, py=0.0, pth=0.0))
    m.update_objects("CMD_MAPOBJ#chair,0.0,2.0")
    m.update_objects("CMD_MAPOBJ#chair,0.0,2.05")     # 5 cm < threshold
    assert not m.objects[0].moving


def test_unknown_distance_object_never_moving():
    m = wm.WorldModel(move_thresh_m=0.2)
    m.update_status(_status(px=0.0, py=0.0))
    m.update_objects("CMD_MAPOBJ#person,0.0,-1.00")
    m.update_objects("CMD_MAPOBJ#person,45.0,-1.00")
    assert not m.objects[0].moving


def test_reset_clears_everything():
    m = wm.WorldModel()
    m.update_status(_status(sonic_cm=100.0, px=1.0, py=1.0))
    m.update_objects("CMD_MAPOBJ#chair,0.0,2.0")
    m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0, px=1.0, py=1.0))
    m.reset()
    assert m.trajectory == [] and m.obstacle_points == [] and m.objects == []
    assert m.foresight_points == [] and m.pose is None and m.goal is None


# ── V-JEPA 2 foresight layer ─────────────────────────────────────────────────

def test_blocked_records_foresight_ahead():
    """A BLOCKED prediction drops a hazard marker at the look-ahead point ahead
    of the robot (bearing 0 → along +Y at heading 0)."""
    m = wm.WorldModel(hazard_lookahead_max_m=2.0)
    m.update_status(_status(wm_label="BLOCKED", clear_dist=1.5, px=0.0, py=0.0, pth=0.0))
    assert len(m.foresight_points) == 1
    hz = m.foresight_points[0]
    assert hz.label == "BLOCKED"
    assert math.isclose(hz.x_m, 0.0, abs_tol=1e-6)
    assert math.isclose(hz.y_m, 1.5, abs_tol=1e-6)   # placed at the depth free-space dist


def test_mixed_records_foresight():
    m = wm.WorldModel()
    m.update_status(_status(wm_label="MIXED", clear_dist=1.0, px=0.0, py=0.0))
    assert len(m.foresight_points) == 1 and m.foresight_points[0].label == "MIXED"


def test_clear_and_unknown_record_no_foresight():
    m = wm.WorldModel()
    m.update_status(_status(wm_label="CLEAR", clear_dist=1.0))
    m.update_status(_status(wm_label="UNKNOWN", clear_dist=1.0))
    assert m.foresight_points == []


def test_foresight_uses_nominal_when_depth_unknown():
    m = wm.WorldModel(hazard_lookahead_m=1.0)
    m.update_status(_status(wm_label="BLOCKED", clear_dist=-1.0, px=0.0, py=0.0, pth=0.0))
    assert math.isclose(m.foresight_points[0].y_m, 1.0, abs_tol=1e-6)


def test_foresight_clamps_large_depth():
    m = wm.WorldModel(hazard_lookahead_max_m=2.0)
    m.update_status(_status(wm_label="BLOCKED", clear_dist=9.0, px=0.0, py=0.0, pth=0.0))
    assert math.isclose(m.foresight_points[0].y_m, 2.0, abs_tol=1e-6)


def test_foresight_projects_with_heading():
    """Facing +X (heading 90°): the look-ahead hazard lands along world +X."""
    m = wm.WorldModel()
    m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0, px=0.0, py=0.0, pth=90.0))
    hz = m.foresight_points[0]
    assert math.isclose(hz.x_m, 1.0, abs_tol=1e-6)
    assert math.isclose(hz.y_m, 0.0, abs_tol=1e-6)


def test_foresight_merges_same_label_nearby():
    """A stationary spot the model keeps flagging must not pile up duplicates."""
    m = wm.WorldModel(foresight_merge_m=0.2)
    for _ in range(5):
        m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0, px=0.0, py=0.0))
    assert len(m.foresight_points) == 1


def test_foresight_keeps_distinct_labels_at_same_spot():
    m = wm.WorldModel(foresight_merge_m=0.2)
    m.update_status(_status(wm_label="MIXED", clear_dist=1.0, px=0.0, py=0.0))
    m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0, px=0.0, py=0.0))
    assert len(m.foresight_points) == 2      # escalation MIXED→BLOCKED both kept


def test_foresight_cap_drops_oldest():
    m = wm.WorldModel(max_foresight=3, foresight_merge_m=0.0)
    for i in range(6):
        m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0,
                                px=float(i), py=0.0))
    assert len(m.foresight_points) == 3


def test_foresight_included_in_bounds():
    m = wm.WorldModel()
    m.update_status(_status(wm_label="BLOCKED", clear_dist=1.0, px=0.0, py=0.0))
    min_x, min_y, max_x, max_y = m.bounds(pad_m=0.5)
    assert max_y >= 1.0                       # the hazard 1 m ahead is enclosed


def test_status_fields_parse_wm_and_clear_dist():
    fields = wm._parse_status_fields(_status(wm_label="blocked", clear_dist=1.25))
    assert fields["wm_label"] == "BLOCKED"    # upper-cased
    assert math.isclose(fields["clear_dist_m"], 1.25)


# ── bounds + colour ─────────────────────────────────────────────────────────

def test_bounds_empty_default():
    assert wm.WorldModel().bounds() == (-1.0, -1.0, 1.0, 1.0)


def test_bounds_encloses_points_with_pad():
    m = wm.WorldModel(trail_step_m=0.0)
    m.update_status(_status(px=0.0, py=0.0))
    m.update_status(_status(px=2.0, py=3.0))
    min_x, min_y, max_x, max_y = m.bounds(pad_m=0.5)
    assert min_x <= -0.5 and min_y <= -0.5
    assert max_x >= 2.5 and max_y >= 3.5


def test_world_to_screen_flips_y():
    """Higher world Y must map to a SMALLER screen y (drawn 'up')."""
    view = (0.0, 0.0, 2.0, 2.0)
    _, sy_low = wm.world_to_screen(1.0, 0.0, view, 100, 100)
    _, sy_high = wm.world_to_screen(1.0, 2.0, view, 100, 100)
    assert sy_high < sy_low


def test_label_color_stable_and_varied():
    assert wm.label_color("person") == wm.label_color("person")
    assert wm.label_color("person") != wm.label_color("chair")
    assert wm.label_color("") == (180, 180, 180)
    r, g, b = wm.label_color("dog")
    assert all(0 <= c <= 255 for c in (r, g, b))
