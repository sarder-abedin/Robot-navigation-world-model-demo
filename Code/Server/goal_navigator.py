"""
goal_navigator.py – Phase 2: track a user-selected goal point across frames.

Given a goal point (normalized image coords from the UI, see CMD_GOAL), follow it
as the scene moves and report:
  • the tracked pixel (so the HUD marker sticks to the object, not the click),
  • the bearing — horizontal offset from image centre, [-1 left .. +1 right] and
    degrees (using a configured camera horizontal FOV),
  • the depth at the goal pixel (metres), sampled from the depth model.

Phase 2 is ANNOTATION ONLY — it does not steer the robot (that is a later phase).

Tracking uses OpenCV's CSRT tracker when available (opencv-contrib-python). It
falls back to normalized-cross-correlation template matching when only base
opencv-python is installed, so it works — with slightly less sticky tracking —
either way and stays unit-testable without the contrib build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoalState:
    active: bool = False           # a goal is set and being tracked
    lost: bool = False             # tracker has lost the goal (no lock for a while)
    reached: bool = False          # the goal is within the arrival distance
    x: float = 0.5                 # tracked goal centre, normalized [0,1]
    y: float = 0.5
    bearing: float = 0.0           # horizontal offset from centre [-1 left .. +1 right]
    bearing_deg: float = 0.0       # bearing in degrees (configured horizontal FOV)
    distance_m: float | None = None  # depth at the goal pixel (m); None if unknown


def goal_steering(base_action, goal_state, center_tol_deg: float = 12.0):
    """Phase 3: turn the safety decision into a goal-directed one.

    Priority is subsumption — **safety always wins**. If the avoidance stack chose
    anything other than a clear-path FORWARD/SLOW (i.e. STOP/REROUTE/BACKUP because
    of an obstacle), we obey it and don't steer toward the goal. Only when the path
    is clear do we point at the goal: spin in place (TURN) toward it until its
    bearing is within center_tol_deg, then drive FORWARD/SLOW toward it.

    Returns (action, turn_direction). turn_direction is "" unless action is TURN.
    A None / inactive / reached / lost goal leaves the action unchanged.
    """
    from decision import Action
    if goal_state is None or not getattr(goal_state, "active", False):
        return base_action, ""
    if goal_state.reached or goal_state.lost:
        return base_action, ""
    # Obstacle avoidance in progress → safety wins, don't chase the goal.
    if base_action not in (Action.FORWARD, Action.SLOW):
        return base_action, ""
    # Path is clear: aim at the goal. Turn in place until it's roughly ahead.
    if abs(goal_state.bearing_deg) > center_tol_deg:
        return Action.TURN, ("right" if goal_state.bearing_deg > 0 else "left")
    return base_action, ""   # goal is ahead → keep driving toward it


def _make_csrt():
    """Return a CSRT tracker instance if this OpenCV build has one, else None."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    return None


def csrt_available() -> bool:
    return _make_csrt() is not None


class _TemplateTracker:
    """Fallback tracker (base opencv-python): NCC template match in a search window.

    Mirrors the cv2 Tracker interface: init(frame, box) and update(frame) →
    (ok, box) with box = (x, y, w, h) in pixels.
    """

    def __init__(self, search_frac: float = 0.30, min_score: float = 0.35):
        self._tmpl = None
        self._box = None
        self._search_frac = search_frac
        self._min_score = min_score

    def init(self, frame_bgr, box) -> bool:
        H, W = frame_bgr.shape[:2]
        x, y, w, h = (int(v) for v in box)
        x = max(0, min(x, W - 2)); y = max(0, min(y, H - 2))
        w = max(8, min(w, W - x)); h = max(8, min(h, H - y))
        self._tmpl = frame_bgr[y:y + h, x:x + w].copy()
        self._box = (x, y, w, h)
        return True

    def update(self, frame_bgr):
        if self._tmpl is None:
            return False, self._box
        H, W = frame_bgr.shape[:2]
        x, y, w, h = self._box
        sw = int(w + 2 * self._search_frac * W)
        sh = int(h + 2 * self._search_frac * H)
        cx, cy = x + w // 2, y + h // 2
        sx = max(0, cx - sw // 2); sy = max(0, cy - sh // 2)
        ex = min(W, sx + sw); ey = min(H, sy + sh)
        window = frame_bgr[sy:ey, sx:ex]
        if window.shape[0] < h or window.shape[1] < w:
            return False, self._box
        res = cv2.matchTemplate(window, self._tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        if maxval < self._min_score:
            return False, self._box
        self._box = (sx + maxloc[0], sy + maxloc[1], w, h)
        return True, self._box


class GoalTracker:
    """Tracks one goal point and derives its bearing + depth for the HUD."""

    def __init__(self, cfg: dict | None = None):
        g = ((cfg or {}).get("goal") or {}) if isinstance(cfg, dict) else {}
        self._patch_frac = float(g.get("patch_frac", 0.12))       # goal patch size (frac of frame)
        self._hfov_deg = float(g.get("horizontal_fov_deg", 66.0))  # camera horizontal FOV
        self._max_lost = int(g.get("max_lost_frames", 15))         # misses before "lost"
        self._arrival_m = float(g.get("arrival_distance_m", 0.4))  # goal depth ≤ this → reached
        self._use_csrt = csrt_available()
        self._tracker = None
        self._pending = None      # (x,y) normalized, awaiting init on the next frame
        self._active = False
        self._lost_count = 0
        self._reached = False     # latched once the goal is within arrival distance
        self._last = GoalState()
        logger.info("GoalTracker using %s tracker",
                    "CSRT (opencv-contrib)" if self._use_csrt else "template-match fallback")

    # ── Control ───────────────────────────────────────────────────────────────

    def set_target(self, x_norm: float, y_norm: float) -> None:
        """Arm a new goal at normalized coords; the tracker inits on the next frame."""
        self._pending = (min(max(x_norm, 0.0), 1.0), min(max(y_norm, 0.0), 1.0))
        self._active = True
        self._lost_count = 0
        self._reached = False

    def clear(self) -> None:
        self._pending = None
        self._tracker = None
        self._active = False
        self._reached = False
        self._last = GoalState()

    @property
    def active(self) -> bool:
        return self._active

    def _new_tracker(self):
        return _make_csrt() or _TemplateTracker()

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, frame_bgr, depth_sampler=None) -> GoalState:
        """Advance tracking one frame. depth_sampler(x_norm,y_norm)->metres|None."""
        if not self._active or frame_bgr is None:
            return GoalState()
        H, W = frame_bgr.shape[:2]

        if self._pending is not None:
            gx, gy = self._pending
            pw = max(16, int(self._patch_frac * W))
            ph = max(16, int(self._patch_frac * H))
            box = (int(gx * W - pw / 2), int(gy * H - ph / 2), pw, ph)
            self._tracker = self._new_tracker()
            try:
                self._tracker.init(frame_bgr, tuple(box))
            except Exception as exc:
                logger.warning("Goal tracker init failed: %s", exc)
                self.clear()
                return GoalState()
            self._pending = None
            self._lost_count = 0
            return self._emit(gx, gy, depth_sampler)

        try:
            ok, box = self._tracker.update(frame_bgr)
        except Exception as exc:
            logger.debug("Goal tracker update error: %s", exc)
            ok, box = False, None

        if ok and box is not None:
            x, y, w, h = box
            self._lost_count = 0
            return self._emit((x + w / 2) / W, (y + h / 2) / H, depth_sampler)

        # Missed this frame: hold the last position; only flag "lost" after a run.
        self._lost_count += 1
        lost = self._lost_count >= self._max_lost
        if lost and not self._last.lost:
            logger.info("Goal lost (no track for %d frames)", self._lost_count)
        self._last = GoalState(
            active=True, lost=lost, reached=self._reached, x=self._last.x, y=self._last.y,
            bearing=self._last.bearing, bearing_deg=self._last.bearing_deg,
            distance_m=self._last.distance_m,
        )
        return self._last

    def _emit(self, cx: float, cy: float, depth_sampler) -> GoalState:
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bearing = 2.0 * (cx - 0.5)                    # -1 left .. +1 right
        bearing_deg = bearing * (self._hfov_deg / 2.0)
        dist = None
        if depth_sampler is not None:
            try:
                d = depth_sampler(cx, cy)
                dist = float(d) if d is not None and d > 0 else None
            except Exception:
                dist = None
        # Arrival: latch once the goal's depth is within the arrival distance, so a
        # noisy depth reading can't un-reach it (cleared only on a new/clear goal).
        if dist is not None and dist <= self._arrival_m:
            self._reached = True
        self._last = GoalState(active=True, lost=False, reached=self._reached, x=cx, y=cy,
                               bearing=bearing, bearing_deg=bearing_deg, distance_m=dist)
        return self._last
