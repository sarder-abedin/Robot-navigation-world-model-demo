"""
ai_viewer.py – Operator UI for the Freenove predictive navigation system.

What this viewer does
─────────────────────
1. Connects to the PC navigation server (CMD port 5003, Video port 8003).
2. Shows live AI state: current action, fused risk bar, V-JEPA 2 prediction
   label, motion pattern, and ultrasonic distance.
3. Lets the operator switch between AUTO (AI drives) and MANUAL (operator drives).
4. In MANUAL mode, provides on-screen drive buttons and arrow-key control.
5. Displays the annotated video stream from the server (Video port 8003).
6. Provides an EMERGENCY STOP kill switch – button AND keyboard shortcuts.

Architecture
────────────
  Pi (client)   → runs YOLO11n + camera + motors; sends CMD_DETECTION
  PC (server)   → runs V-JEPA 2 + SSv2 + decision; listens on 5003/8003
  This viewer   → connects to PC on 5003/8003; shows AI state + video

Control modes
─────────────
  AUTO MODE    → PC decision fuser drives the robot via CMD_AIMOVE
  MANUAL MODE  → Operator drives via arrow keys / buttons (CMD_MOTOR relayed by PC)

TCP protocol (send)
───────────────────
  CMD_AIMODE#0          → stop AI, halt motors (STOP AI / MANUAL transition)
  CMD_AIMODE#1          → switch to baseline reactive mode
  CMD_AIMODE#2          → switch to predictive mode
  CMD_MOTOR#<L>#<R>     → manual motor command (−4095 … +4095)
  CMD_KILL#0            → stop AI + robot, shut down server process

TCP protocol (receive)
──────────────────────
  CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>

Kill switch controls
────────────────────
  Space / Escape           → EMERGENCY STOP  (motors stop, AI disabled)
  Ctrl+Q or the button     → SHUTDOWN SERVER (stops the entire demo process)
"""

from __future__ import annotations

import math
import os
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_map     # noqa: E402  (framework-agnostic egocentric-map geometry + parsing)
import world_map   # noqa: E402  (framework-agnostic world-anchored trajectory map)

# ── Colour maps ────────────────────────────────────────────────────────────────
ACTION_CSS = {
    "FORWARD":  "background:#1a7a1a; color:white;",
    "SLOW":     "background:#c8841a; color:white;",
    "STOP":     "background:#8b0000; color:white;",
    "REROUTE":  "background:#7a3a00; color:white;",
    "BACKUP":   "background:#a0521a; color:white;",
    "TURN":     "background:#1a5a7a; color:white;",
    "---":      "background:#444;    color:#aaa;",
}
WM_CSS = {
    "BLOCKED": "color:#ff4444;",
    "MIXED":   "color:#ffaa44;",
    "CLEAR":   "color:#44cc44;",
    "UNKNOWN": "color:#aaaaaa;",
}

# Kill-switch button styles
_KILL_READY  = ("background:#cc0000; color:white; font-size:15px; "
                "font-weight:bold; padding:12px; border-radius:4px;")
_KILL_SENT   = ("background:#550000; color:#ff9999; font-size:15px; "
                "font-weight:bold; padding:12px; border-radius:4px;")
_SHUTDOWN_READY = ("background:#4a0000; color:#ffbbbb; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")
_SHUTDOWN_SENT  = ("background:#220000; color:#ff6666; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")

_AUTO_BTN_ACTIVE   = "background:#0a5a8a; color:white; font-weight:bold; padding:6px;"
_AUTO_BTN_INACTIVE = "background:#2a2a2a; color:#aaa; font-weight:bold; padding:6px;"
_MANUAL_BTN_ACTIVE   = "background:#5a3a00; color:white; font-weight:bold; padding:6px;"
_MANUAL_BTN_INACTIVE = "background:#2a2a2a; color:#aaa; font-weight:bold; padding:6px;"

_DRIVE_BTN = ("background:#333; color:white; font-size:16px; "
              "font-weight:bold; padding:10px; border-radius:4px; min-width:60px;")
_DRIVE_STOP_BTN = ("background:#550000; color:white; font-size:14px; "
                   "font-weight:bold; padding:10px; border-radius:4px; min-width:60px;")

CMD_PORT   = 5003
VIDEO_PORT = 8003
MAX_FRAME_BYTES = 10 * 1024 * 1024   # reject a desynced/garbage frame length

SPEED_FULL = 1500
SPEED_SLOW = 600

# Action → robot-glyph colour on the map (mirrors ACTION_CSS so the two views match).
ACTION_RGB = {
    "FORWARD": (26, 122, 26), "SLOW": (200, 132, 26), "STOP": (170, 20, 20),
    "REROUTE": (122, 58, 0), "BACKUP": (160, 82, 26), "TURN": (26, 90, 122),
    "---": (110, 110, 110),
}
GOAL_RGB = {
    "tracking": (60, 200, 255), "lost": (200, 200, 60),
    "reached": (60, 220, 90), "none": (110, 110, 110),
}


class NavMapWidget(QWidget):
    """Top-down egocentric navigation map (robot fixed at bottom-centre, facing up).

    Draws the current depth free-space (LEFT/CENTER/RIGHT), the ultrasonic reading,
    the chosen clear direction, and the goal by bearing + distance — all relative to
    the robot *right now* (there is no odometry, so nothing is world-anchored). All
    geometry comes from nav_map (unit-tested); this widget only paints it.
    """

    def __init__(self, max_range_m: float = 5.0, fov_deg: float = 66.0, parent=None):
        super().__init__(parent)
        self._model = nav_map.MapModel(max_range_m=max_range_m, fov_deg=fov_deg)
        self._max_range_m = max_range_m
        self._fov_deg = fov_deg
        self.setMinimumSize(260, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setToolTip("Local (egocentric) navigation map — robot at bottom, facing up. "
                        "Not a world map (no odometry).")

    def set_model(self, model: "nav_map.MapModel") -> None:
        self._model = model
        self.update()

    # ── painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        m = self._model
        w, h = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(0, 0, w, h, QColor(18, 18, 20))

        margin = 26.0
        origin = (w / 2.0, h - margin)
        ox, oy = origin
        ppm = nav_map.fit_scale(w, h, margin, m.max_range_m, m.fov_deg)

        self._draw_rings(p, origin, ppm, m)
        self._draw_fov(p, origin, ppm, m)
        self._draw_sectors(p, origin, ppm, m)
        self._draw_ultrasonic(p, origin, ppm, m)
        self._draw_goal(p, origin, ppm, m)
        self._draw_robot(p, origin, m)

        # Caption
        p.setPen(QColor(140, 140, 150))
        p.setFont(QFont("sans", 7))
        p.drawText(6, 12, "LOCAL MAP · egocentric (no odometry)")
        p.end()

    def _draw_rings(self, p, origin, ppm, m):
        ox, oy = origin
        p.setFont(QFont("sans", 7))
        for r in range(1, int(m.max_range_m) + 1):
            rad = r * ppm
            p.setPen(QtGui.QPen(QColor(55, 55, 62), 1, Qt.DotLine))
            # upper half-circle only (the robot looks forward/up)
            p.drawArc(QtCore.QRectF(ox - rad, oy - rad, 2 * rad, 2 * rad), 0, 180 * 16)
            p.setPen(QColor(90, 90, 100))
            p.drawText(QtCore.QPointF(ox + 3, oy - rad + 2), f"{r} m")

    def _draw_fov(self, p, origin, ppm, m):
        p.setPen(QtGui.QPen(QColor(70, 70, 80), 1, Qt.DashLine))
        for edge in (-m.fov_deg / 2.0, m.fov_deg / 2.0):
            x, y = nav_map.polar_to_world(m.max_range_m, edge)
            sx, sy = nav_map.world_to_screen(x, y, origin, ppm)
            p.drawLine(QtCore.QPointF(*origin), QtCore.QPointF(sx, sy))

    def _draw_sectors(self, p, origin, ppm, m):
        ox, oy = origin
        dists = {"LEFT": m.depth_left_m, "CENTER": m.depth_center_m, "RIGHT": m.depth_right_m}
        for name, d in dists.items():
            start, end = nav_map.sector_edge_bearings(m.fov_deg, name)
            known = d is not None and d > 0
            reach = d if known else m.max_range_m
            rad = reach * ppm
            r, g, b = nav_map.proximity_color(d if known else None)
            # Free-space pie slice for the third, out to the measured obstacle.
            # Qt angles: 0° = +x (right), CCW, in 1/16°. Screen y is down, and our
            # bearing is clockwise-from-up, so on-screen angle = 90 - bearing.
            a0 = 90.0 - end
            span = end - start
            fill = QColor(r, g, b, 70 if known else 30)
            p.setBrush(fill)
            p.setPen(Qt.NoPen)
            p.drawPie(QtCore.QRectF(ox - rad, oy - rad, 2 * rad, 2 * rad),
                      int(round(a0 * 16)), int(round(span * 16)))
            if known:
                # Obstacle boundary arc at the measured distance.
                p.setBrush(Qt.NoBrush)
                p.setPen(QtGui.QPen(QColor(r, g, b), 2.4))
                p.drawArc(QtCore.QRectF(ox - rad, oy - rad, 2 * rad, 2 * rad),
                          int(round(a0 * 16)), int(round(span * 16)))
                # Highlight the chosen open direction.
                if m.clear_dir == name:
                    cb = nav_map.sector_center_bearings(m.fov_deg)[name]
                    x, y = nav_map.polar_to_world(reach * 0.6, cb)
                    sx, sy = nav_map.world_to_screen(x, y, origin, ppm)
                    p.setPen(QColor(60, 220, 90))
                    p.setBrush(QColor(60, 220, 90))
                    p.drawEllipse(QtCore.QPointF(sx, sy), 3, 3)

    def _draw_ultrasonic(self, p, origin, ppm, m):
        if m.sonic_m is None:
            return
        x, y = nav_map.polar_to_world(min(m.sonic_m, m.max_range_m), 0.0)
        sx, sy = nav_map.world_to_screen(x, y, origin, ppm)
        col = QColor(255, 70, 70) if m.sonic_m < 0.25 else QColor(90, 200, 220)
        p.setPen(QtGui.QPen(col, 2))
        p.setBrush(col)
        p.drawEllipse(QtCore.QPointF(sx, sy), 3.5, 3.5)
        p.setFont(QFont("sans", 7))
        p.drawText(QtCore.QPointF(sx + 6, sy), f"sonar {m.sonic_m:.2f}m")

    def _draw_goal(self, p, origin, ppm, m):
        if not m.has_goal:
            return
        r, g, b = GOAL_RGB.get(m.goal_status, GOAL_RGB["none"])
        col = QColor(r, g, b)
        # If the depth at the goal is unknown, draw it at the ring edge as a bearing hint.
        d = m.goal_dist_m if (m.goal_dist_m and m.goal_dist_m > 0) else m.max_range_m
        x, y = nav_map.polar_to_world(min(d, m.max_range_m), m.goal_bearing_deg)
        sx, sy = nav_map.world_to_screen(x, y, origin, ppm)
        p.setPen(QtGui.QPen(col, 1.5, Qt.DashLine))
        p.drawLine(QtCore.QPointF(*origin), QtCore.QPointF(sx, sy))
        p.setPen(QtGui.QPen(col, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(sx, sy), 6, 6)
        p.drawLine(QtCore.QPointF(sx - 8, sy), QtCore.QPointF(sx + 8, sy))
        p.drawLine(QtCore.QPointF(sx, sy - 8), QtCore.QPointF(sx, sy + 8))
        p.setFont(QFont("sans", 7))
        p.drawText(QtCore.QPointF(sx + 9, sy - 6), f"goal ({m.goal_status})")

    def _draw_robot(self, p, origin, m):
        ox, oy = origin
        r, g, b = ACTION_RGB.get(m.action, ACTION_RGB["---"])
        p.setBrush(QColor(r, g, b))
        p.setPen(QtGui.QPen(QColor(230, 230, 235), 1.2))
        tri = QtGui.QPolygonF([
            QtCore.QPointF(ox, oy - 12), QtCore.QPointF(ox - 8, oy + 6),
            QtCore.QPointF(ox + 8, oy + 6),
        ])
        p.drawPolygon(tri)
        p.setPen(QColor(230, 230, 235))
        p.setFont(QFont("sans", 8, QFont.Bold))
        p.drawText(QtCore.QPointF(ox + 12, oy + 6), m.action)


class WorldMapWidget(QWidget):
    """World-anchored, ACCUMULATING navigation map (the 'random-walk' style view).

    Anchored to the robot's dead-reckoned pose (shipped in CMD_AISTATUS), it grows
    a trajectory trail plus a scatter of ultrasonic obstacles and YOLO objects in a
    single fixed world frame — where the robot has been and what it has seen, not
    just 'right now'. Moving objects are drawn distinctly. All geometry/accumulation
    lives in world_map (unit-tested); this widget only paints and auto-fits the view.

    The pose is open-loop (no odometry) → the whole map DRIFTS; it's a best-effort
    local world view, not a survey-grade global map.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._m = world_map.WorldModel()
        self.setMinimumSize(260, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setToolTip("World-anchored trajectory map (dead-reckoning, no odometry "
                        "→ drifts). Trail = path; red dots = ultrasonic; squares = YOLO "
                        "objects (hollow = moving); diamonds = V-JEPA 2 predicted hazards "
                        "(red BLOCKED / amber MIXED).")

    def update_status(self, line: str) -> None:
        self._m.update_status(line)
        self.update()

    def update_objects(self, line: str) -> None:
        self._m.update_objects(line)
        self.update()

    def reset(self) -> None:
        self._m.reset()
        self.update()

    # ── painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, _ev):
        m = self._m
        w, h = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(0, 0, w, h, QColor(18, 18, 20))

        view = m.bounds(pad_m=0.5)

        def to_screen(x, y):
            return world_map.world_to_screen(x, y, view, w, h)

        self._draw_grid(p, view, w, h, to_screen)
        self._draw_trajectory(p, m, to_screen)
        self._draw_foresight(p, m, to_screen)    # V-JEPA 2 predictions (under the live layers)
        self._draw_obstacles(p, m, to_screen)
        self._draw_objects(p, m, to_screen)
        self._draw_goal(p, m, to_screen)
        self._draw_robot(p, m, to_screen)

        # Caption + stats.
        p.setPen(QColor(140, 140, 150))
        p.setFont(QFont("sans", 7))
        p.drawText(6, 12, "WORLD MAP · dead-reckoned (drifts, no odometry)")
        p.setPen(QColor(110, 110, 120))
        p.drawText(6, h - 6, f"path {len(m.trajectory)} · obstacles "
                             f"{len(m.obstacle_points)} · objects {len(m.objects)} · "
                             f"V-JEPA2 {len(m.foresight_points)}")
        p.end()

    def _draw_grid(self, p, view, w, h, to_screen):
        # Grid lines for a sense of scale. Guard against a non-finite or huge span
        # (a long/drifted run) so we never call math.floor(nan) or spin a loop
        # drawing millions of lines: skip if non-finite, and widen the spacing so
        # the count stays bounded (~1 m normally, coarser as the extent grows).
        min_x, min_y, max_x, max_y = view
        span = max(max_x - min_x, max_y - min_y)
        if not math.isfinite(span) or span <= 0:
            return
        step = 1.0 if span <= 12.0 else math.ceil(span / 12.0)
        p.setPen(QtGui.QPen(QColor(38, 38, 44), 1, Qt.DotLine))
        p.setFont(QFont("sans", 6))
        x0 = math.floor(min_x / step) * step
        while x0 <= max_x:
            sx, _ = to_screen(x0, min_y)
            p.drawLine(QtCore.QPointF(sx, 0), QtCore.QPointF(sx, h))
            x0 += step
        y0 = math.floor(min_y / step) * step
        while y0 <= max_y:
            _, sy = to_screen(min_x, y0)
            p.drawLine(QtCore.QPointF(0, sy), QtCore.QPointF(w, sy))
            y0 += step

    def _draw_trajectory(self, p, m, to_screen):
        if len(m.trajectory) < 2:
            return
        pts = [QtCore.QPointF(*to_screen(pp.x_m, pp.y_m)) for pp in m.trajectory]
        # Fading trail: older segments dimmer.
        n = len(pts)
        for i in range(1, n):
            t = i / n
            col = QColor(60, int(120 + 90 * t), int(200 + 40 * t), int(60 + 180 * t))
            p.setPen(QtGui.QPen(col, 2.0))
            p.drawLine(pts[i - 1], pts[i])
        # Origin marker (where the run started).
        p.setPen(QtGui.QPen(QColor(120, 120, 130), 1))
        p.setBrush(Qt.NoBrush)
        ox, oy = to_screen(m.trajectory[0].x_m, m.trajectory[0].y_m)
        p.drawEllipse(QtCore.QPointF(ox, oy), 4, 4)

    def _draw_foresight(self, p, m, to_screen):
        # V-JEPA 2 predicted-hazard markers: translucent diamonds ahead of the
        # path, coloured by the world-model label (matches the WM_CSS panel:
        # BLOCKED red, MIXED amber). This is the PREDICTIVE layer — where the
        # world model foresaw trouble — distinct from the reactive ones.
        colors = {"BLOCKED": QColor(255, 68, 68, 150), "MIXED": QColor(255, 170, 68, 130)}
        edges = {"BLOCKED": QColor(255, 68, 68), "MIXED": QColor(255, 170, 68)}
        for hz in m.foresight_points:
            sx, sy = to_screen(hz.x_m, hz.y_m)
            p.setBrush(colors.get(hz.label, QColor(180, 180, 180, 110)))
            p.setPen(QtGui.QPen(edges.get(hz.label, QColor(180, 180, 180)), 1))
            d = 5.0
            p.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(sx, sy - d), QtCore.QPointF(sx + d, sy),
                QtCore.QPointF(sx, sy + d), QtCore.QPointF(sx - d, sy),
            ]))

    def _draw_obstacles(self, p, m, to_screen):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(230, 70, 70, 190))
        for (x, y) in m.obstacle_points:
            sx, sy = to_screen(x, y)
            p.drawEllipse(QtCore.QPointF(sx, sy), 2.6, 2.6)

    def _draw_objects(self, p, m, to_screen):
        p.setFont(QFont("sans", 7))
        for o in m.objects:
            sx, sy = to_screen(o.x_m, o.y_m)
            r, g, b = world_map.label_color(o.label)
            col = QColor(r, g, b)
            if o.moving:
                # Moving obstacle: hollow, thicker outline (stands out from statics).
                p.setBrush(Qt.NoBrush)
                p.setPen(QtGui.QPen(col, 2.4))
                p.drawEllipse(QtCore.QPointF(sx, sy), 7, 7)
                p.drawEllipse(QtCore.QPointF(sx, sy), 3, 3)
            else:
                p.setPen(QtGui.QPen(QColor(235, 235, 240), 1))
                p.setBrush(col if o.dist_known else QColor(r, g, b, 90))
                p.drawRect(QtCore.QRectF(sx - 4, sy - 4, 8, 8))
            tag = o.label + (" ▶" if o.moving else "")
            p.setPen(col.lighter(140))
            p.drawText(QtCore.QPointF(sx + 8, sy + 3), tag)

    def _draw_goal(self, p, m, to_screen):
        if m.goal is None:
            return
        sx, sy = to_screen(*m.goal)
        col = QColor(60, 220, 90)
        p.setPen(QtGui.QPen(col, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(sx, sy), 7, 7)
        p.drawLine(QtCore.QPointF(sx - 9, sy), QtCore.QPointF(sx + 9, sy))
        p.drawLine(QtCore.QPointF(sx, sy - 9), QtCore.QPointF(sx, sy + 9))
        p.setFont(QFont("sans", 7))
        p.drawText(QtCore.QPointF(sx + 10, sy - 6), "goal")

    def _draw_robot(self, p, m, to_screen):
        if m.pose is None:
            return
        sx, sy = to_screen(m.pose.x_m, m.pose.y_m)
        # Heading: 0° = +Y (up), increasing clockwise (turning right). Screen y is
        # down, so a heading θ points at screen-angle (−90°+θ) from +x.
        th = math.radians(m.pose.heading_deg)
        fx, fy = math.sin(th), -math.cos(th)         # forward on screen (+Y up)
        rx, ry = math.cos(th), math.sin(th)          # right on screen
        tip = QtCore.QPointF(sx + fx * 11, sy + fy * 11)
        bl = QtCore.QPointF(sx - fx * 6 - rx * 6, sy - fy * 6 - ry * 6)
        br = QtCore.QPointF(sx - fx * 6 + rx * 6, sy - fy * 6 + ry * 6)
        p.setBrush(QColor(90, 170, 255))
        p.setPen(QtGui.QPen(QColor(235, 235, 240), 1.3))
        p.drawPolygon(QtGui.QPolygonF([tip, bl, br]))


class AIViewer(QMainWindow):
    # Qt signal so the network recv thread can safely update the UI thread
    status_received = pyqtSignal(str)
    mapobj_received = pyqtSignal(str)     # CMD_MAPOBJ line → world map (GUI thread)
    frame_received = pyqtSignal(object)   # QImage from the video thread → main thread
    disconnected = pyqtSignal()           # worker thread requests teardown on the GUI thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Freenove Predictive Navigation – AI Viewer")
        self.resize(1180, 800)   # wide two-column layout (visuals left, controls right)

        self._cmd_sock: socket.socket | None = None
        self._video_sock: socket.socket | None = None
        self._connected = False
        self._recv_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None

        self._control_mode: str = "AUTO"   # "AUTO" | "MANUAL"
        self._ai_active_mode: int = 0      # last CMD_AIMODE sent: 0 idle, 1 baseline, 2 predictive
        self._manual_speed: int = SPEED_FULL
        self._keys_held: set[int] = set()  # avoid repeated motor sends on auto-repeat
        self._goal_selected: bool = False  # a goal has been set (needed for Goal-Following)
        self._nav_mode = None              # None (not picked yet) | "avoid" | "goal"
        self._fps_times: list[float] = []  # video-frame arrival times for the FPS readout

        self._build_ui()
        self._register_shortcuts()
        self._update_activation_gate()   # AI activation disabled until connected + goal set

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_status_bar)
        self._ui_timer.start(200)

        self.status_received.connect(self._process_status)
        self.mapobj_received.connect(self._process_mapobj)
        self.frame_received.connect(self._show_frame)   # GUI-thread pixmap update
        self.disconnected.connect(self._disconnect)     # GUI-thread teardown

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Compact TWO-COLUMN layout — visuals (video + maps) on the left, all the
        # controls on the right — so the whole UI fits at the default window size
        # without scrolling (the old single tall column forced a scrollbar). The
        # scroll area is kept only as a fallback for very small windows; a minimum
        # size keeps the layout from collapsing into an unusable state.
        self.setMinimumSize(900, 560)
        central = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.setCentralWidget(scroll)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Quick-start strip: the three steps to drive, so it's obvious what to do
        steps = QLabel("① Connect   →   ② pick a Navigation Mode   →   "
                       "③ AUTO (AI drives) or MANUAL (you drive)")
        steps.setStyleSheet("color:#cfd8dc; background:#2a2f33; border-radius:4px; "
                            "padding:5px 8px; font-size:11px;")
        steps.setAlignment(Qt.AlignCenter)
        steps.setWordWrap(True)
        root.addWidget(steps)

        # ── Connection row ────────────────────────────────────────────────────
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("PC Server IP:"))
        self._ip_edit = QLineEdit("127.0.0.1")
        self._ip_edit.setFixedWidth(145)
        conn_row.addWidget(self._ip_edit)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setFixedWidth(90)
        self._btn_connect.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self._btn_connect)
        conn_row.addStretch()
        root.addLayout(conn_row)

        # ── Two-column body: visuals (video + maps) on the LEFT, all controls on
        # the RIGHT, so the whole UI fits on a normal screen without scrolling. ──
        body = QHBoxLayout()
        body.setSpacing(10)
        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        body.addLayout(left_col, 1)     # visuals take the extra width (maps grow)
        body.addLayout(right_col, 0)    # controls stay compact
        root.addLayout(body, 1)

        # ── Video display ─────────────────────────────────────────────────────
        self._video_label = QLabel("[ No video – connect to server ]")
        self._video_label.setFixedSize(400, 300)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "background:#1a1a1a; border:1px solid #555; color:#666;"
        )
        # Click-to-set-goal: capture clicks on the video and map them to normalized
        # image coords. Active only while "Set Goal" is armed.
        self._video_label.mousePressEvent = self._on_video_click
        self._pix_w = 0   # size of the last displayed pixmap (for click→image mapping)
        self._pix_h = 0

        # Left column: video on top, then the local + world maps SIDE BY SIDE
        # below it (was a tall stacked layout that forced scrolling). The maps get
        # a COMPACT fixed height so they don't balloon into big (mostly-black)
        # boxes when the window is tall.
        self._map = NavMapWidget()
        self._map.setFixedHeight(250)
        self._world_map = WorldMapWidget()
        self._world_map.setFixedHeight(250)

        vid_row = QHBoxLayout()
        vid_row.addStretch(1)
        vid_row.addWidget(self._video_label)
        vid_row.addStretch(1)
        left_col.addLayout(vid_row)

        maps_row = QHBoxLayout()
        maps_row.setSpacing(6)
        maps_row.addWidget(self._map, 1)          # local (egocentric)
        maps_row.addWidget(self._world_map, 1)    # world-anchored trajectory
        left_col.addLayout(maps_row)

        map_row = QHBoxLayout()
        self._chk_map = QCheckBox("Show local map")
        self._chk_map.setChecked(True)
        self._chk_map.setToolTip("Top-down egocentric map (robot-centred; no odometry, so it's a live local view).")
        self._chk_map.toggled.connect(self._map.setVisible)
        self._chk_world = QCheckBox("Show world map")
        self._chk_world.setChecked(True)
        self._chk_world.setToolTip("World-anchored accumulating trajectory + obstacle map "
                                   "(dead-reckoning, no odometry → drifts).")
        self._chk_world.toggled.connect(self._world_map.setVisible)
        self._btn_reset_map = QPushButton("Reset trail")
        self._btn_reset_map.setToolTip("Clear the accumulated world map and start a fresh trajectory.")
        self._btn_reset_map.clicked.connect(self._world_map.reset)
        # V-JEPA 2 dense-feature view: side-by-side PCA panel in the video HUD.
        self._chk_featviz = QCheckBox("V-JEPA 2 view")
        self._chk_featviz.setToolTip(
            "Show what V-JEPA 2 sees beside the camera: a PCA visualization of its "
            "dense patch features (on by default; switch off to reclaim video width)."
        )
        self._chk_featviz.setChecked(True)
        self._chk_featviz.toggled.connect(self._send_featureviz)
        map_row.addStretch(1)
        map_row.addWidget(self._chk_map)
        map_row.addWidget(self._chk_world)
        map_row.addWidget(self._chk_featviz)
        map_row.addWidget(self._btn_reset_map)
        map_row.addStretch(1)
        left_col.addLayout(map_row)
        left_col.addStretch(1)     # keep the visuals top-aligned (gray gap, not big maps)

        # ── Navigation Mode: pick one on connect (nothing pre-selected) ───────
        navmode_box = QGroupBox("Navigation Mode")
        navmode_col = QVBoxLayout(navmode_box)
        navmode_row = QHBoxLayout()
        self._radio_avoid = QRadioButton("Obstacle Avoidance")
        self._radio_avoid.setToolTip("AI avoids obstacles (predictive/baseline). No goal needed.")
        self._radio_goal = QRadioButton("Goal Following")
        self._radio_goal.setToolTip("Set a goal on the video; the robot steers to it (avoidance still overrides).")
        # No default: the operator must choose. Act only on the newly-selected radio.
        self._radio_avoid.toggled.connect(lambda on: self._on_nav_mode_changed("avoid") if on else None)
        self._radio_goal.toggled.connect(lambda on: self._on_nav_mode_changed("goal") if on else None)
        navmode_row.addWidget(self._radio_avoid)
        navmode_row.addWidget(self._radio_goal)
        navmode_col.addLayout(navmode_row)
        self._navmode_hint = QLabel("Pick a mode to enable AI (MANUAL works now).")
        self._navmode_hint.setAlignment(Qt.AlignCenter)
        self._navmode_hint.setStyleSheet("color:#e0a000;")
        navmode_col.addWidget(self._navmode_hint)
        right_col.addWidget(navmode_box)

        # ── Goal point (Goal-Following mode only) ─────────────────────────────
        self._goal_box = QGroupBox("Navigation Goal")
        goal_col = QVBoxLayout(self._goal_box)
        goal_row = QHBoxLayout()
        self._btn_set_goal = QPushButton("🎯  Set Goal (click video)")
        self._btn_set_goal.setCheckable(True)
        self._btn_set_goal.setToolTip(
            "Arm goal selection, then click a point on the video. The server tracks "
            "it and shows its bearing/depth on the HUD. Does not steer the robot yet."
        )
        goal_row.addWidget(self._btn_set_goal)
        self._btn_clear_goal = QPushButton("Clear Goal")
        self._btn_clear_goal.clicked.connect(self._clear_goal)
        goal_row.addWidget(self._btn_clear_goal)
        goal_col.addLayout(goal_row)
        self._goal_status_lbl = QLabel("")
        self._goal_status_lbl.setAlignment(Qt.AlignCenter)
        goal_col.addWidget(self._goal_status_lbl)
        right_col.addWidget(self._goal_box)
        self._goal_box.setVisible(False)   # shown only in Goal-Following mode

        # ── AI state panel ────────────────────────────────────────────────────
        state_box = QGroupBox("AI State")
        grid = QtWidgets.QGridLayout(state_box)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self._action_label = self._make_action_label("---")
        self._risk_bar = QProgressBar()
        self._risk_bar.setRange(0, 100)
        self._risk_bar.setTextVisible(True)
        self._risk_bar.setFormat("Risk: %p%")
        self._wm_val    = self._make_info_val("UNKNOWN")
        self._pat_val   = self._make_info_val("UNKNOWN")
        self._sonic_val = self._make_info_val("---")
        self._depth_val = self._make_info_val("---")
        self._fps_val   = self._make_info_val("---")
        self._ssv2_val  = self._make_info_val("---")
        self._ssv2_val.setWordWrap(True)

        # Two label/value pairs per row so the panel is short (was 8 stacked rows).
        grid.addWidget(QLabel("Action:"),   0, 0); grid.addWidget(self._action_label, 0, 1)
        grid.addWidget(QLabel("Risk:"),     0, 2); grid.addWidget(self._risk_bar,     0, 3)
        grid.addWidget(QLabel("V-JEPA 2:"), 1, 0); grid.addWidget(self._wm_val,       1, 1)
        grid.addWidget(QLabel("Motion:"),   1, 2); grid.addWidget(self._pat_val,      1, 3)
        grid.addWidget(QLabel("Sonic:"),    2, 0); grid.addWidget(self._sonic_val,    2, 1)
        grid.addWidget(QLabel("Depth:"),    2, 2); grid.addWidget(self._depth_val,    2, 3)
        grid.addWidget(QLabel("FPS:"),      3, 0); grid.addWidget(self._fps_val,      3, 1)
        # SSv2 caption is long → span the full width on its own row.
        grid.addWidget(QLabel("SSv2:"),     4, 0); grid.addWidget(self._ssv2_val,     4, 1, 1, 3)

        right_col.addWidget(state_box)

        # ── AI Model (predictive vs baseline) ─────────────────────────────────
        mode_box = QGroupBox("AI Model  (AUTO)")
        mode_row = QHBoxLayout(mode_box)

        self._btn_predictive = QPushButton("PREDICTIVE")
        self._btn_predictive.setStyleSheet(
            "background:#1a5f1a; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_predictive.setToolTip("Switch to predictive mode (V-JEPA 2 active)")
        self._btn_predictive.clicked.connect(lambda: self._send_ai_mode(2))
        mode_row.addWidget(self._btn_predictive)

        self._btn_baseline = QPushButton("BASELINE")
        self._btn_baseline.setStyleSheet(
            "background:#7a5500; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_baseline.setToolTip("Switch to baseline reactive mode (no V-JEPA 2)")
        self._btn_baseline.clicked.connect(lambda: self._send_ai_mode(1))
        mode_row.addWidget(self._btn_baseline)

        right_col.addWidget(mode_box)

        # ── Run logging (server-side) ─────────────────────────────────────────
        # Toggles CSV + annotated-frame logging on the PC (CMD_LOGGING#1|0). The
        # data is written on the server (logs_rpi/), not on this UI machine.
        log_box = QGroupBox("Run Logging")
        log_row = QHBoxLayout(log_box)
        self._chk_logging = QCheckBox("Record run log (on by default)")
        self._chk_logging.setToolTip(
            "Server-side run logging (logs_rpi/ on the PC running main_server.py). "
            "ON by default; untick to stop recording this run."
        )
        # Checked before wiring the signal so it doesn't fire _send_logging while
        # disconnected (which would revert it); the real CMD_LOGGING is sent on connect.
        self._chk_logging.setChecked(True)
        self._chk_logging.toggled.connect(self._send_logging)
        log_row.addWidget(self._chk_logging)
        right_col.addWidget(log_box)

        # ── Drive Control ─────────────────────────────────────────────────────
        drive_box = QGroupBox("Drive")
        drive_layout = QVBoxLayout(drive_box)
        drive_layout.setSpacing(4)

        # AUTO / MANUAL toggle row — the two buttons share the width equally and
        # never shrink below a readable size.
        toggle_row = QHBoxLayout()
        self._btn_auto = QPushButton("AUTO  ·  AI drives")
        self._btn_auto.setStyleSheet(_AUTO_BTN_ACTIVE)
        self._btn_auto.setMinimumHeight(36)
        self._btn_auto.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_auto.setToolTip("Let the AI decision fuser control the robot (Ctrl+A)")
        self._btn_auto.clicked.connect(self._switch_to_auto)
        toggle_row.addWidget(self._btn_auto)

        self._btn_manual = QPushButton("MANUAL  ·  you drive")
        self._btn_manual.setStyleSheet(_MANUAL_BTN_INACTIVE)
        self._btn_manual.setMinimumHeight(36)
        self._btn_manual.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_manual.setToolTip("Take direct control via buttons or arrow keys (Ctrl+M)")
        self._btn_manual.clicked.connect(self._switch_to_manual)
        toggle_row.addWidget(self._btn_manual)
        drive_layout.addLayout(toggle_row)

        # Manual panel (shown only in MANUAL mode): a fixed-size D-pad, CENTRED so it
        # keeps its size and position no matter how the window is resized.
        self._drive_widget = QWidget()
        manual_col = QVBoxLayout(self._drive_widget)
        manual_col.setSpacing(6)
        manual_col.setContentsMargins(0, 0, 0, 0)

        dpad = QWidget()
        dg = QtWidgets.QGridLayout(dpad)
        dg.setSpacing(6)
        dg.setContentsMargins(0, 0, 0, 0)
        self._btn_fwd  = self._make_drive_btn("▲", lambda: self._drive_press("FWD"))
        self._btn_back = self._make_drive_btn("▼", lambda: self._drive_press("BACK"))
        self._btn_left = self._make_drive_btn("◄", lambda: self._drive_press("LEFT"))
        self._btn_right = self._make_drive_btn("►", lambda: self._drive_press("RIGHT"))
        self._btn_drive_stop = QPushButton("■")
        self._btn_drive_stop.setStyleSheet(_DRIVE_STOP_BTN)
        self._btn_drive_stop.setFixedSize(50, 42)
        self._btn_drive_stop.setToolTip("Stop motors")
        self._btn_drive_stop.clicked.connect(self._drive_stop)
        dg.addWidget(self._btn_fwd,        0, 1)
        dg.addWidget(self._btn_left,       1, 0)
        dg.addWidget(self._btn_drive_stop, 1, 1)
        dg.addWidget(self._btn_right,      1, 2)
        dg.addWidget(self._btn_back,       2, 1)
        dpad_row = QHBoxLayout()
        dpad_row.addStretch(1)
        dpad_row.addWidget(dpad)
        dpad_row.addStretch(1)
        manual_col.addLayout(dpad_row)

        # Speed selector (centred under the D-pad)
        speed_row = QHBoxLayout()
        speed_row.addStretch(1)
        speed_row.addWidget(QLabel("Speed:"))
        self._radio_full = QRadioButton("Full")
        self._radio_full.setChecked(True)
        self._radio_full.toggled.connect(self._update_speed)
        self._radio_slow = QRadioButton("Slow")
        speed_row.addWidget(self._radio_full)
        speed_row.addWidget(self._radio_slow)
        speed_row.addStretch(1)
        manual_col.addLayout(speed_row)

        drive_layout.addWidget(self._drive_widget)
        self._drive_widget.setVisible(False)  # hidden in AUTO mode

        right_col.addWidget(drive_box)

        # ── Kill switch panel ─────────────────────────────────────────────────
        kill_box = QGroupBox("Kill Switch")
        kill_box.setStyleSheet(
            "QGroupBox { border:2px solid #aa0000; border-radius:4px; "
            "margin-top:6px; font-weight:bold; color:#ff6666; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }"
        )
        kill_layout = QVBoxLayout(kill_box)
        kill_layout.setSpacing(4)

        self._btn_kill = QPushButton("EMERGENCY STOP   [Space / Esc]")
        self._btn_kill.setStyleSheet(_KILL_READY)
        self._btn_kill.setToolTip(
            "Immediately stops all motors and disables AI navigation.\n"
            "Server stays running; resume with PREDICTIVE or BASELINE.\n"
            "Keyboard: Space or Escape"
        )
        self._btn_kill.clicked.connect(self._emergency_stop)
        kill_layout.addWidget(self._btn_kill)

        self._btn_shutdown = QPushButton("SHUTDOWN SERVER   [Ctrl+Q]")
        self._btn_shutdown.setStyleSheet(_SHUTDOWN_READY)
        self._btn_shutdown.setToolTip(
            "Stops motors AND shuts down the server process.\n"
            "Use this to end the demo completely.\n"
            "Keyboard: Ctrl+Q"
        )
        self._btn_shutdown.clicked.connect(self._shutdown_server)
        kill_layout.addWidget(self._btn_shutdown)
        # (The shortcuts are already on the button labels + the status bar, so the
        # extra hint line is dropped to keep the panel short.)

        right_col.addWidget(kill_box)
        right_col.addStretch(1)      # keep the controls top-aligned

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QLabel("Not connected – enter the PC server IP and click Connect")
        self._status_bar.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self._status_bar)

    def _make_action_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Monospace", 13, QFont.Bold))
        lbl.setStyleSheet(ACTION_CSS.get(text, ACTION_CSS["---"]))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumWidth(180)
        return lbl

    def _make_info_val(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Monospace", 11))
        lbl.setStyleSheet("color:#aaa;")
        return lbl

    def _make_drive_btn(self, symbol: str, slot) -> QPushButton:
        btn = QPushButton(symbol)
        btn.setStyleSheet(_DRIVE_BTN)
        btn.setFixedSize(50, 42)   # fixed so the D-pad never shrinks/reflows on resize
        btn.pressed.connect(slot)
        btn.released.connect(self._drive_stop)
        return btn

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────

    def _register_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space),  self, self._emergency_stop)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._emergency_stop)
        QShortcut(QKeySequence("Ctrl+Q"), self, self._shutdown_server)
        QShortcut(QKeySequence("Ctrl+P"), self, lambda: self._send_ai_mode(2))
        QShortcut(QKeySequence("Ctrl+B"), self, lambda: self._send_ai_mode(1))
        QShortcut(QKeySequence("Ctrl+A"), self, self._switch_to_auto)
        QShortcut(QKeySequence("Ctrl+M"), self, self._switch_to_manual)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._control_mode != "MANUAL" or event.isAutoRepeat():
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            if key not in self._keys_held:
                self._keys_held.add(key)
                self._apply_key_drive()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._control_mode != "MANUAL" or event.isAutoRepeat():
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self._keys_held.discard(key)
            if not self._keys_held:
                self._send_motor(0, 0)
            else:
                self._apply_key_drive()
        else:
            super().keyReleaseEvent(event)

    def _apply_key_drive(self) -> None:
        """Send motor command based on the set of currently-held arrow keys."""
        s = self._manual_speed
        if Qt.Key_Up in self._keys_held and Qt.Key_Left in self._keys_held:
            self._send_motor(s // 2, s)
        elif Qt.Key_Up in self._keys_held and Qt.Key_Right in self._keys_held:
            self._send_motor(s, s // 2)
        elif Qt.Key_Down in self._keys_held and Qt.Key_Left in self._keys_held:
            self._send_motor(-s, -s // 2)
        elif Qt.Key_Down in self._keys_held and Qt.Key_Right in self._keys_held:
            self._send_motor(-s // 2, -s)
        elif Qt.Key_Up in self._keys_held:
            self._send_motor(s, s)
        elif Qt.Key_Down in self._keys_held:
            self._send_motor(-s, -s)
        elif Qt.Key_Left in self._keys_held:
            self._send_motor(-s, s)
        elif Qt.Key_Right in self._keys_held:
            self._send_motor(s, -s)

    # ── Drive button actions ───────────────────────────────────────────────────

    def _drive_press(self, direction: str) -> None:
        s = self._manual_speed
        mapping = {
            "FWD":   (s, s),
            "BACK":  (-s, -s),
            "LEFT":  (-s, s),
            "RIGHT": (s, -s),
        }
        L, R = mapping.get(direction, (0, 0))
        self._send_motor(L, R)

    def _drive_stop(self) -> None:
        if not self._keys_held:  # don't interrupt keyboard driving
            self._send_motor(0, 0)

    def _update_speed(self) -> None:
        self._manual_speed = SPEED_FULL if self._radio_full.isChecked() else SPEED_SLOW

    # ── Drive control mode ─────────────────────────────────────────────────────

    def _switch_to_auto(self) -> None:
        self._control_mode = "AUTO"
        self._keys_held.clear()
        self._drive_widget.setVisible(False)
        self._btn_auto.setStyleSheet(_AUTO_BTN_ACTIVE)
        self._btn_manual.setStyleSheet(_MANUAL_BTN_INACTIVE)
        # Resume predictive AI
        self._send_ai_mode(2)
        self._status_bar.setText("AUTO MODE – AI decision fuser is driving")

    def _switch_to_manual(self) -> None:
        self._control_mode = "MANUAL"
        self._drive_widget.setVisible(True)
        self._btn_auto.setStyleSheet(_AUTO_BTN_INACTIVE)
        self._btn_manual.setStyleSheet(_MANUAL_BTN_ACTIVE)
        # Disable AI motors; PC relays manual CMD_MOTOR from now on
        self._send_ai_mode(0)
        self._send_motor(0, 0)
        self._status_bar.setText(
            "MANUAL MODE – arrow keys / buttons drive  |  Ctrl+A = return to AUTO"
        )

    # ── Kill switch actions ────────────────────────────────────────────────────

    def _emergency_stop(self) -> None:
        self._send_ai_mode(0)
        self._send_motor(0, 0)
        self._btn_kill.setText("STOPPED  ✓  – click PREDICTIVE / BASELINE to resume")
        self._btn_kill.setStyleSheet(_KILL_SENT)
        self._status_bar.setText(
            "EMERGENCY STOP sent – motors halted, AI disabled. "
            "Click PREDICTIVE or BASELINE to resume."
        )
        QTimer.singleShot(4000, self._reset_kill_button)

    def _shutdown_server(self) -> None:
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot send shutdown command.")
            return
        try:
            self._cmd_sock.sendall(b"CMD_KILL#0\n")
            self._btn_shutdown.setText("SHUTDOWN SENT  ✓")
            self._btn_shutdown.setStyleSheet(_SHUTDOWN_SENT)
            self._status_bar.setText(
                "SHUTDOWN command sent – server is stopping. "
                "Connection will drop in a moment."
            )
        except Exception as exc:
            self._status_bar.setText(f"Shutdown send error: {exc}")

    def _reset_kill_button(self) -> None:
        self._btn_kill.setText("EMERGENCY STOP   [Space / Esc]")
        self._btn_kill.setStyleSheet(_KILL_READY)

    # ── Connection ─────────────────────────────────────────────────────────────

    def _toggle_connection(self) -> None:
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        ip = self._ip_edit.text().strip()
        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.settimeout(3.0)
            self._cmd_sock.connect((ip, CMD_PORT))
            self._cmd_sock.settimeout(None)
            self._connected = True
            self._btn_connect.setText("Disconnect")
            self._status_bar.setText(
                f"Connected to {ip}:{CMD_PORT}  |  "
                "Space/Esc = stop   Ctrl+Q = shutdown   Ctrl+M = manual"
            )
            self._recv_thread = threading.Thread(
                target=self._recv_loop, args=(self._cmd_sock,), daemon=True, name="CmdRecv"
            )
            self._recv_thread.start()

            # Start IDLE with NO mode picked: the operator chooses a Navigation
            # Mode first, then explicitly activates. Manual driving is available now.
            self._goal_selected = False
            self._clear_nav_mode()
            self._send_ai_mode(0)
            # Run logging is ON by default — push the checkbox state to the server.
            self._send_logging(self._chk_logging.isChecked())
            # V-JEPA 2 feature view is ON by default — push its state too.
            self._send_featureviz(self._chk_featviz.isChecked())
            self._update_activation_gate()
            self._status_bar.setText("Connected – pick a Navigation Mode to begin (or drive MANUAL).")

            # Video connection (non-fatal if unavailable)
            self._video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._video_sock.settimeout(3.0)
                self._video_sock.connect((ip, VIDEO_PORT))
                self._video_sock.settimeout(1.0)
                self._video_thread = threading.Thread(
                    target=self._video_loop, args=(self._video_sock,), daemon=True, name="VideoRecv"
                )
                self._video_thread.start()
            except Exception:
                try:
                    self._video_sock.close()
                except Exception:
                    pass
                self._video_sock = None
                self._video_label.setText("[ Video unavailable ]")

        except Exception as exc:
            try:
                if self._cmd_sock:
                    self._cmd_sock.close()
            except Exception:
                pass
            self._cmd_sock = None
            self._status_bar.setText(f"Connection failed: {exc}")

    def _disconnect(self) -> None:
        self._connected = False
        self._keys_held.clear()
        for s in (self._cmd_sock, self._video_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._cmd_sock = self._video_sock = None
        self._goal_selected = False
        self._clear_nav_mode()             # require a fresh mode pick on reconnect
        self._update_activation_gate()     # re-lock AI activation
        self._btn_connect.setText("Connect")
        self._video_label.setText("[ No video – connect to server ]")
        self._status_bar.setText("Disconnected")

    # ── Network threads ────────────────────────────────────────────────────────

    def _recv_loop(self, sock) -> None:
        # Bind to the socket this thread was started with; if a reconnect swaps
        # self._cmd_sock, this stale thread stops instead of clobbering the new one.
        buf = ""
        while self._connected and sock is self._cmd_sock:
            try:
                raw = sock.recv(1024)
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("CMD_AISTATUS"):
                        self.status_received.emit(line)
                    elif line.startswith("CMD_MAPOBJ"):
                        self.mapobj_received.emit(line)
            except Exception:
                break
        if sock is self._cmd_sock:            # only tear down the CURRENT connection
            self.disconnected.emit()          # marshalled to the GUI thread

    def _video_loop(self, sock) -> None:
        while self._connected and sock is self._video_sock:
            header = self._recv_exact(sock, 4)
            if header is None:
                break
            n = struct.unpack("<I", header)[0]
            if not (0 < n <= MAX_FRAME_BYTES):   # desynced/garbage length → give up
                break
            jpg = self._recv_exact(sock, n)
            if jpg is None:
                break
            try:
                arr = np.frombuffer(jpg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                # .copy() so the QImage owns its buffer (rgb goes out of scope) and
                # is safe to hand to the GUI thread, which builds the QPixmap.
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
                self.frame_received.emit(qimg)
            except Exception:
                continue

    def _show_frame(self, qimg) -> None:
        """GUI-thread slot: turn the QImage into a scaled pixmap."""
        pix = QPixmap.fromImage(qimg).scaled(
            420, 315, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._pix_w = pix.width()
        self._pix_h = pix.height()
        self._video_label.setPixmap(pix)
        # Video FPS (moved off the HUD): count arrivals over a ~1 s window.
        import time as _t
        now = _t.monotonic()
        self._fps_times.append(now)
        while self._fps_times and now - self._fps_times[0] > 1.0:
            self._fps_times.pop(0)
        if len(self._fps_times) >= 2:
            self._fps_val.setText(f"{len(self._fps_times) - 1:d}")

    # ── Goal selection (Phase 1: send CMD_GOAL, server draws the marker) ─────────

    def _on_video_click(self, event) -> None:
        """Map a click on the video to normalized image coords and send CMD_GOAL.

        Only acts while 'Set Goal' is armed. The displayed pixmap is centred in the
        label with KeepAspectRatio, so account for any letterbox offset.
        """
        if not self._btn_set_goal.isChecked():
            return
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot set goal.")
            return
        if self._pix_w <= 0 or self._pix_h <= 0:
            self._status_bar.setText("No video yet – cannot set goal.")
            return
        off_x = (self._video_label.width() - self._pix_w) / 2.0
        off_y = (self._video_label.height() - self._pix_h) / 2.0
        nx = (event.pos().x() - off_x) / self._pix_w
        ny = (event.pos().y() - off_y) / self._pix_h
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            self._status_bar.setText("Click inside the video image to set a goal.")
            return
        # Per-mille integers: the server's message parser is integer-only.
        try:
            self._cmd_sock.sendall(
                f"CMD_GOAL#{int(nx * 1000)}#{int(ny * 1000)}\n".encode("utf-8")
            )
            self._status_bar.setText(
                f"Goal set at ({nx:.2f}, {ny:.2f}) – now activate AI to start."
            )
            self._goal_selected = True
            self._update_activation_gate()     # AI activation now unlocked
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")
        self._btn_set_goal.setChecked(False)   # one-shot; re-arm for the next goal

    def _clear_goal(self) -> None:
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot clear goal.")
            return
        try:
            self._cmd_sock.sendall(b"CMD_GOAL_CLEAR\n")
            # Clearing the goal re-locks AI activation and returns the robot to idle.
            self._cmd_sock.sendall(b"CMD_AIMODE#0\n")
            self._ai_active_mode = 0   # idled: don't let a later nav-mode switch auto-resume
            self._goal_selected = False
            self._update_activation_gate()
            self._status_bar.setText("Goal cleared – AI idle until a new goal is set.")
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")

    def _recv_exact(self, sock, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                # The pipeline can be slow to emit the first frame; keep waiting as
                # long as we're still the active connection instead of dying.
                if self._connected and sock is self._video_sock:
                    continue
                return None
            except Exception:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    # ── Status display ─────────────────────────────────────────────────────────

    def _process_status(self, line: str) -> None:
        # CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>
        #   #<ssv2>#<clear_dist_m>#<clear_dir>#<goal_status>
        #   #<depth_left_m>#<depth_right_m>#<goal_bearing_deg>#<goal_dist_m>
        #   (trailing fields optional — old servers omit them)
        # Feed the 2D navigation map (parsing is shared with nav_map, unit-tested).
        self._map.set_model(nav_map.parse_status(line))
        # Feed the world-anchored trajectory map (pose + sonar + goal accumulate).
        self._world_map.update_status(line)
        parts = line.split("#")
        if len(parts) < 6:
            return
        _, action, risk_pct, wm_label, pattern, sonic = parts[:6]
        ssv2 = parts[6] if len(parts) > 6 else ""
        clear_dist = parts[7] if len(parts) > 7 else ""
        clear_dir = parts[8] if len(parts) > 8 else ""
        goal_status = parts[9] if len(parts) > 9 else "none"
        self._apply_goal_status(goal_status)

        self._action_label.setText(action)
        self._action_label.setStyleSheet(ACTION_CSS.get(action, ACTION_CSS["---"]))

        try:
            pct = int(risk_pct)
            self._risk_bar.setValue(pct)
            r = min(pct * 2, 255)
            g = min((100 - pct) * 2, 255)
            self._risk_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: rgb({r},{g},0); }}"
            )
        except ValueError:
            pass

        self._wm_val.setText(wm_label)
        self._wm_val.setStyleSheet(WM_CSS.get(wm_label, WM_CSS["UNKNOWN"]))

        self._pat_val.setText(pattern)

        sonic = sonic.strip()
        try:
            cm = float(sonic)
            if cm < 0:
                # -1 = no echo / sensor blind — don't render it as a 0 cm obstacle.
                self._sonic_val.setText("--- (no echo)")
                self._sonic_val.setStyleSheet("color:#888;")
            else:
                self._sonic_val.setText(f"{cm:.1f} cm")
                self._sonic_val.setStyleSheet(
                    "color:#ff4444;" if cm < 20 else "color:#44cc44;"
                )
        except ValueError:
            self._sonic_val.setText(sonic)

        # Depth + SSv2 (moved off the video HUD into this panel).
        try:
            dm = float(clear_dist)
            self._depth_val.setText(
                f"{dm:.2f} m ahead  (open: {clear_dir or '?'})" if dm >= 0 else "---"
            )
        except ValueError:
            self._depth_val.setText("---")
        self._ssv2_val.setText(ssv2 or "---")

    def _process_mapobj(self, line: str) -> None:
        # CMD_MAPOBJ#<label>,<bearing>,<dist>;… — YOLO objects for the world map
        # (parsing + world projection shared with world_map, unit-tested).
        self._world_map.update_objects(line)

    def _update_status_bar(self) -> None:
        if self._connected:
            mode_hint = "AUTO" if self._control_mode == "AUTO" else "MANUAL (↑↓←→)"
            self._status_bar.setText(
                f"Connected  [{mode_hint}]  |  "
                "Space/Esc = emergency stop   Ctrl+Q = shutdown   Ctrl+M/A = mode"
            )

    # ── Command helpers ────────────────────────────────────────────────────────

    def _send_ai_mode(self, mode: int) -> None:
        """Send CMD_AIMODE#<mode> to the server."""
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot send command.")
            return
        # Activating AI (baseline/predictive) requires a picked mode (+ a goal for
        # Goal Following). Idle (0) is always allowed. Guards the keyboard shortcuts.
        if mode in (1, 2) and not self._ai_activation_allowed():
            self._status_bar.setText(
                "Pick a Navigation Mode first"
                + (" and set a goal." if self._nav_mode == "goal" else ".")
            )
            return
        try:
            self._cmd_sock.sendall(f"CMD_AIMODE#{mode}\n".encode("utf-8"))
            self._ai_active_mode = mode   # remember so a nav-mode switch can resume it
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")

    def _apply_goal_status(self, status: str) -> None:
        """Reflect the server's goal status (none|tracking|lost|reached) in the UI."""
        if status == "reached":
            # The server has stopped the robot on arrival; it waits for a new
            # command (set a new goal, or re-activate AI).
            self._goal_status_lbl.setText(
                "✅ GOAL REACHED — robot stopped.\nSet a new goal or re-activate AI."
            )
            self._goal_status_lbl.setStyleSheet("color:#33cc33; font-weight:bold;")
        elif status == "lost":
            self._goal_status_lbl.setText("⚠ Goal lost — re-select it on the video.")
            self._goal_status_lbl.setStyleSheet("color:#e0a000;")
        elif status == "tracking":
            self._goal_status_lbl.setText("Tracking goal…")
            self._goal_status_lbl.setStyleSheet("color:#33aaff;")
        else:
            self._goal_status_lbl.setText("")

    def _clear_nav_mode(self) -> None:
        """Deselect both mode radios (autoExclusive won't let us uncheck normally)."""
        for r in (self._radio_avoid, self._radio_goal):
            r.setAutoExclusive(False)
            r.setChecked(False)
            r.setAutoExclusive(True)
        self._nav_mode = None
        self._goal_box.setVisible(False)

    def _on_nav_mode_changed(self, mode: str) -> None:
        """Operator picked a mode. Set it up but do NOT start driving — the robot
        only moves after an explicit AI activation (and, for Goal Following, once a
        goal is set). Obstacle Avoidance needs no goal.
        """
        self._nav_mode = mode
        self._goal_box.setVisible(mode == "goal")
        kept_driving = False
        if self._connected and self._cmd_sock:
            try:
                self._cmd_sock.sendall(
                    f"CMD_GOALFOLLOW#{1 if mode == 'goal' else 0}\n".encode("utf-8"))
                # If the AI was already driving (AUTO + a live model), keep it
                # driving across the switch instead of idling the robot — switching
                # nav mode mid-run used to send CMD_AIMODE#0 and stop everything.
                # Only continue when the new mode is actually allowed (Goal Following
                # needs a goal); otherwise idle safely.
                was_driving = self._control_mode == "AUTO" and self._ai_active_mode in (1, 2)
                if was_driving and self._ai_activation_allowed():
                    self._send_ai_mode(self._ai_active_mode)   # resume same model, new nav mode
                    kept_driving = True
                else:
                    self._cmd_sock.sendall(b"CMD_AIMODE#0\n")   # stay idle until explicit start
                    self._ai_active_mode = 0
            except Exception as exc:
                self._status_bar.setText(f"Send error: {exc}")
        self._update_activation_gate()
        if kept_driving:
            self._status_bar.setText(
                ("Goal Following" if mode == "goal" else "Obstacle Avoidance")
                + " – AI still driving.")
        else:
            self._status_bar.setText(
                "Goal Following – set a goal on the video, then activate AI."
                if mode == "goal" else
                "Obstacle Avoidance – click PREDICTIVE or BASELINE to start."
            )

    def _ai_activation_allowed(self) -> bool:
        """AI can be started only once connected AND a mode is picked; Goal
        Following additionally needs a goal."""
        if not self._connected or self._nav_mode is None:
            return False
        if self._nav_mode == "goal":
            return self._goal_selected
        return True

    def _update_activation_gate(self) -> None:
        """Reflect _ai_activation_allowed() on the AUTO/PREDICTIVE/BASELINE buttons.
        MANUAL driving stays available regardless (even before a mode is picked)."""
        armed = self._ai_activation_allowed()
        if armed:
            tip = ""
        elif not self._connected:
            tip = "Connect to the server first."
        elif self._nav_mode is None:
            tip = "Pick a Navigation Mode to begin."
        else:
            tip = "Set a goal on the video first, then activate AI."
        for btn in (self._btn_auto, self._btn_predictive, self._btn_baseline):
            btn.setEnabled(armed)
            if tip:
                btn.setToolTip(tip)
        # The "pick a mode" hint shows until a mode is chosen.
        self._navmode_hint.setVisible(self._nav_mode is None)

    def _send_logging(self, on: bool) -> None:
        """Toggle server-side run logging (CMD_LOGGING#1|0)."""
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot toggle logging.")
            # revert the checkbox to reflect that nothing was sent
            self._chk_logging.blockSignals(True)
            self._chk_logging.setChecked(False)
            self._chk_logging.blockSignals(False)
            return
        try:
            self._cmd_sock.sendall(f"CMD_LOGGING#{1 if on else 0}\n".encode("utf-8"))
            self._status_bar.setText(
                f"Server run logging {'ENABLED' if on else 'disabled'}"
            )
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")

    def _send_featureviz(self, on: bool) -> None:
        """Toggle the V-JEPA 2 dense-feature HUD panel (CMD_FEATUREVIZ#1|0)."""
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot toggle V-JEPA 2 view.")
            # revert the checkbox to reflect that nothing was sent
            self._chk_featviz.blockSignals(True)
            self._chk_featviz.setChecked(False)
            self._chk_featviz.blockSignals(False)
            return
        try:
            self._cmd_sock.sendall(f"CMD_FEATUREVIZ#{1 if on else 0}\n".encode("utf-8"))
            self._status_bar.setText(
                f"V-JEPA 2 feature view {'ENABLED' if on else 'disabled'}"
            )
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")

    def _send_motor(self, left: int, right: int) -> None:
        """Send CMD_MOTOR#<L>#<R> to the server (relayed to robot in manual mode)."""
        if not (self._cmd_sock and self._connected):
            return
        try:
            self._cmd_sock.sendall(f"CMD_MOTOR#{left}#{right}\n".encode("utf-8"))
        except Exception as exc:
            self._status_bar.setText(f"Motor cmd error: {exc}")

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._disconnect()
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window,          QColor(45,  45,  45))
    palette.setColor(QtGui.QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base,            QColor(30,  30,  30))
    palette.setColor(QtGui.QPalette.AlternateBase,   QColor(50,  50,  50))
    palette.setColor(QtGui.QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button,          QColor(55,  55,  55))
    palette.setColor(QtGui.QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Highlight,       QColor(0,   120, 215))
    palette.setColor(QtGui.QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = AIViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
