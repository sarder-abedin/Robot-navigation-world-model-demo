"""
world_map.py – framework-agnostic core of the world-anchored navigation map.

Unlike nav_map.py (which is egocentric / "right now"), this map ACCUMULATES over
time, anchored to the robot's dead-reckoned pose (Code/Server/pose_estimator.py,
shipped in CMD_AISTATUS): the robot's trajectory, the ultrasonic-detected
obstacles, and the YOLO objects are all projected into one fixed world frame and
kept, so the display grows into a trail + a scatter of obstacle/object points —
the "random-walk" style map the operator asked for.

No PyQt / no numpy → unit-testable (tests_rpi/test_world_map_rpi.py). The
QPainter widget (ai_viewer.WorldMapWidget) is the thin drawing layer on top.

Because the pose is open-loop dead-reckoning (no odometry), the whole map drifts;
it is a best-effort *local* world view, not a survey-grade global map.

World frame (identical to pose_estimator's):
  • origin (0,0) = the robot's start, +Y = initial forward, +X = initial right,
  • heading_deg 0 = facing +Y, increasing when turning RIGHT (toward +X),
  • a robot-relative reading is (bearing_deg from forward: +right/−left, dist_m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# CMD_AISTATUS trailing pose fields (appended after the map/goal fields; see
# ai_pipeline._broadcast_status). Older servers omit them → parse returns None.
#   … 13 goal_dist_m  14 pose_x_m  15 pose_y_m  16 pose_heading_deg
_POSE_X_IDX, _POSE_Y_IDX, _POSE_TH_IDX = 14, 15, 16


@dataclass
class WorldPose:
    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0


@dataclass
class WorldObject:
    """A YOLO detection projected into the world frame."""
    label: str
    x_m: float
    y_m: float
    dist_known: bool = True    # False → depth was unknown, position is a fallback estimate
    moving: bool = False       # world position shifted between frames (ego-motion removed)


@dataclass
class WorldModel:
    """Accumulated world state the widget paints. Mutated in place by update_*()."""
    trajectory: list = field(default_factory=list)       # [WorldPose, …] robot path
    obstacle_points: list = field(default_factory=list)  # [(x,y), …] ultrasonic hits
    objects: list = field(default_factory=list)          # [WorldObject, …] latest frame
    pose: WorldPose | None = None
    goal: tuple | None = None                            # (x, y) | None
    # Caps + thresholds (keep memory bounded; decimate near-duplicate points).
    max_trail: int = 4000
    max_obstacles: int = 4000
    trail_step_m: float = 0.02        # don't record a path point until moved this far
    obstacle_merge_m: float = 0.08    # merge an ultrasonic hit into a nearby existing one
    move_thresh_m: float = 0.18       # world shift over one frame to call an object "moving"
    _prev_objects: list = field(default_factory=list)    # last frame's known-dist objects

    # ── mutation ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.trajectory.clear()
        self.obstacle_points.clear()
        self.objects.clear()
        self._prev_objects.clear()
        self.pose = None
        self.goal = None

    def update_status(self, line: str) -> None:
        """Consume a CMD_AISTATUS line: advance the pose/trajectory, add the
        straight-ahead ultrasonic hit, and place the goal (all in world coords)."""
        pose = parse_pose(line)
        if pose is None:
            return
        self.pose = pose
        self._append_trail(pose)

        st = _parse_status_fields(line)
        # Ultrasonic hit: straight ahead (bearing 0) at the sonar range → world.
        if st["sonic_m"] is not None:
            xr, yr = polar_to_robot(st["sonic_m"], 0.0)
            self._add_obstacle(*robot_to_world(xr, yr, pose))
        # Goal marker (bearing + distance) → world, when tracking with a range.
        if st["goal_has"] and st["goal_dist_m"] is not None:
            xr, yr = polar_to_robot(st["goal_dist_m"], st["goal_bearing_deg"])
            self.goal = robot_to_world(xr, yr, pose)
        elif not st["goal_has"]:
            self.goal = None

    def update_objects(self, line: str, fallback_dist_m: float = 1.5) -> None:
        """Consume a CMD_MAPOBJ line: project each YOLO object to the world frame
        (using the latest pose) and flag the ones that moved since last frame."""
        if self.pose is None:
            return
        parsed = parse_mapobj(line)
        objs: list = []
        for label, bearing_deg, dist_m in parsed:
            known = dist_m is not None and dist_m > 0
            d = dist_m if known else fallback_dist_m
            xr, yr = polar_to_robot(d, bearing_deg)
            x, y = robot_to_world(xr, yr, self.pose)
            objs.append(WorldObject(label=label, x_m=x, y_m=y, dist_known=known))
        _flag_moving(objs, self._prev_objects, self.move_thresh_m)
        self.objects = objs
        # Only known-distance objects seed next frame's motion comparison.
        self._prev_objects = [o for o in objs if o.dist_known]

    # ── helpers ─────────────────────────────────────────────────────────────
    def _append_trail(self, pose: WorldPose) -> None:
        if self.trajectory:
            last = self.trajectory[-1]
            if math.hypot(pose.x_m - last.x_m, pose.y_m - last.y_m) < self.trail_step_m:
                # Still update the heading of the current spot (in-place turns) so
                # the robot marker rotates without piling up path points.
                last.heading_deg = pose.heading_deg
                return
        self.trajectory.append(WorldPose(pose.x_m, pose.y_m, pose.heading_deg))
        if len(self.trajectory) > self.max_trail:
            del self.trajectory[0:len(self.trajectory) - self.max_trail]

    def _add_obstacle(self, x: float, y: float) -> None:
        # Merge into a nearby existing hit so a stationary wall doesn't accumulate
        # thousands of overlapping points (keeps the scatter and memory bounded).
        for px, py in self.obstacle_points[-64:]:   # only scan the recent tail (cheap)
            if math.hypot(x - px, y - py) < self.obstacle_merge_m:
                return
        self.obstacle_points.append((x, y))
        if len(self.obstacle_points) > self.max_obstacles:
            del self.obstacle_points[0:len(self.obstacle_points) - self.max_obstacles]

    def bounds(self, pad_m: float = 0.5) -> tuple:
        """(min_x, min_y, max_x, max_y) enclosing everything, for auto-fit."""
        xs, ys = [], []
        for p in self.trajectory:
            xs.append(p.x_m); ys.append(p.y_m)
        for (x, y) in self.obstacle_points:
            xs.append(x); ys.append(y)
        for o in self.objects:
            xs.append(o.x_m); ys.append(o.y_m)
        if self.pose is not None:
            xs.append(self.pose.x_m); ys.append(self.pose.y_m)
        if self.goal is not None:
            xs.append(self.goal[0]); ys.append(self.goal[1])
        if not xs:
            return (-1.0, -1.0, 1.0, 1.0)
        return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


# ── parsing ─────────────────────────────────────────────────────────────────

def _pos_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def parse_pose(line: str) -> WorldPose | None:
    """Read the trailing pose fields from a CMD_AISTATUS line, or None if the
    server is too old to send them (backward-compatible)."""
    parts = line.strip().split("#")
    if not parts or parts[0] != "CMD_AISTATUS" or len(parts) <= _POSE_TH_IDX:
        return None
    try:
        return WorldPose(
            x_m=float(parts[_POSE_X_IDX]),
            y_m=float(parts[_POSE_Y_IDX]),
            heading_deg=float(parts[_POSE_TH_IDX]),
        )
    except (TypeError, ValueError):
        return None


def _parse_status_fields(line: str) -> dict:
    """Pull the sonar + goal fields the world map needs from a CMD_AISTATUS line."""
    parts = line.strip().split("#")
    out = {"sonic_m": None, "goal_has": False,
           "goal_bearing_deg": 0.0, "goal_dist_m": None}
    if len(parts) < 6 or parts[0] != "CMD_AISTATUS":
        return out
    son = _pos_float(parts[5])                    # cm on the wire
    out["sonic_m"] = (son / 100.0) if son is not None else None
    if len(parts) > 9:
        status = (parts[9] or "none").strip().lower()
        out["goal_has"] = status in ("tracking", "lost", "reached")
    if len(parts) > 12:
        try:
            out["goal_bearing_deg"] = float(parts[12])
        except (TypeError, ValueError):
            out["goal_bearing_deg"] = 0.0
    if len(parts) > 13:
        out["goal_dist_m"] = _pos_float(parts[13])
    return out


def parse_mapobj(line: str) -> list:
    """Parse CMD_MAPOBJ#<label>,<bearing_deg>,<dist_m>;… → [(label, bearing, dist|None)].

    dist is None when the server sent ≤0 (depth unknown for that object).
    Tolerant of blanks and malformed triples (they're skipped, not raised)."""
    line = line.strip()
    if not line.startswith("CMD_MAPOBJ"):
        return []
    _, _, payload = line.partition("#")
    out = []
    for chunk in payload.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = chunk.split(",")
        if len(fields) < 3:
            continue
        label = fields[0].strip()
        try:
            bearing = float(fields[1])
            dist = float(fields[2])
        except (TypeError, ValueError):
            continue
        out.append((label, bearing, dist if dist > 0 else None))
    return out


# ── geometry ────────────────────────────────────────────────────────────────

def polar_to_robot(dist_m: float, bearing_deg: float) -> tuple:
    """(distance, bearing°) → (x_right_m, y_forward_m) in the robot frame."""
    th = math.radians(bearing_deg)
    return dist_m * math.sin(th), dist_m * math.cos(th)


def robot_to_world(x_r: float, y_r: float, pose: WorldPose) -> tuple:
    """Robot-frame (right, forward) metres → world (x, y), given the robot pose.

    forward_world = (sinθ, cosθ), right_world = (cosθ, −sinθ); world offset is
    x_r·right_world + y_r·forward_world, added to the robot's world position."""
    th = math.radians(pose.heading_deg)
    s, c = math.sin(th), math.cos(th)
    x = pose.x_m + x_r * c + y_r * s
    y = pose.y_m - x_r * s + y_r * c
    return x, y


def world_to_screen(x_m: float, y_m: float, view: tuple,
                    width_px: float, height_px: float, margin_px: float = 8.0) -> tuple:
    """World metres → screen pixels for a given view box (min_x,min_y,max_x,max_y),
    preserving aspect ratio and centring, with +Y up on screen."""
    min_x, min_y, max_x, max_y = view
    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)
    ppm = min((width_px - 2 * margin_px) / span_x,
              (height_px - 2 * margin_px) / span_y)
    ppm = max(1.0, ppm)
    # Centre the (usually non-square) content within the widget.
    cx_world = (min_x + max_x) / 2.0
    cy_world = (min_y + max_y) / 2.0
    sx = width_px / 2.0 + (x_m - cx_world) * ppm
    sy = height_px / 2.0 - (y_m - cy_world) * ppm    # +Y up
    return sx, sy


def _flag_moving(cur: list, prev: list, thresh_m: float) -> None:
    """Mark a current object 'moving' if the nearest same-label object in the
    previous frame sits > thresh_m away in the WORLD frame. Both positions are
    pose-anchored, so the robot's own motion is already removed — what's left is
    the object's real displacement (plus a little pose drift)."""
    for o in cur:
        if not o.dist_known:
            continue
        best = None
        for p in prev:
            if p.label != o.label:
                continue
            d = math.hypot(o.x_m - p.x_m, o.y_m - p.y_m)
            if best is None or d < best:
                best = d
        if best is not None and best > thresh_m:
            o.moving = True


# Stable colour per YOLO class label (deterministic hash → pleasant hue).
def label_color(label: str) -> tuple:
    """Deterministic RGB for a class label so the same object keeps its colour."""
    if not label:
        return (180, 180, 180)
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    r, g, b = _hsv_to_rgb(hue, 0.55, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]
