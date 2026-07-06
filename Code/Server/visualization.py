"""
visualization.py – OpenCV HUD overlay for the navigation pipeline.

Annotates a BGR frame (for display / streaming) with:
  • Obstacle bounding boxes (colour-coded by class)
  • Fused risk bar (green→yellow→red)
  • Current action label
  • V-JEPA 2 prediction label (world model)
  • Temporal motion pattern label (SSv2-style)
  • Navigation mode badge (PREDICTIVE / BASELINE)
  • Ultrasonic distance
  • FPS counter

The annotated frame is returned as a BGR numpy array and can be
  - shown in an OpenCV window (Pi with display)
  - JPEG-encoded and sent to the TCP video client
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

# BGR colour palette for obstacle classes
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "person":        (0, 165, 255),
    "chair":         (255,   0, 128),
    "couch":         (128,   0, 255),
    "dining table":  (255, 128,   0),
    "potted plant":  (  0, 200, 100),
    "suitcase":      (200, 200,   0),
    "backpack":      (  0, 200, 200),
    "bottle":        (100, 100, 255),
    "book":          (180,  80, 180),
}
DEFAULT_COLOUR = (180, 180, 180)

ACTION_COLOURS = {
    "FORWARD": ( 0, 220,   0),
    "SLOW":    ( 0, 200, 255),
    "STOP":    ( 0,   0, 220),
    "REROUTE": ( 0, 100, 255),
    "BACKUP":  ( 0, 165, 255),   # orange
}

WM_COLOURS = {
    "BLOCKED": ( 0,   0, 200),
    "CLEAR":   ( 0, 200,   0),
    "MIXED":   ( 0, 180, 255),
    "UNKNOWN": (150, 150, 150),
}


class Visualizer:
    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        vis = cfg["visualization"]
        self._show = vis.get("show_window", False)
        self._win = vis.get("window_name", "Predictive Navigation")
        self._overlay_det = vis.get("overlay_detections", True)
        self._overlay_risk = vis.get("overlay_risk_bar", True)
        self._overlay_action = vis.get("overlay_action", True)
        # These text overlays are shown in the UI panel BELOW the video, so they
        # default OFF on the video itself to keep the image uncluttered. The
        # spatial cues (detection boxes, depth L/C/R bars, goal marker+arrow) and
        # Action/Risk/Goal-readout stay on the video.
        self._overlay_wm = vis.get("overlay_world_model_label", False)   # V-JEPA2 + motion text
        self._overlay_sonic = vis.get("overlay_sonic", False)
        self._overlay_fps = vis.get("overlay_fps", False)
        self._overlay_mode_badge = vis.get("overlay_mode_badge", False)
        self._overlay_ssv2 = vis.get("overlay_ssv2", False)
        self._overlay_depth_text = vis.get("overlay_depth_text", False)  # keep the L/C/R bars
        self._stream_annotated = vis.get("stream_annotated", True)
        self._mode = navigation_mode

        self._fps_buf: deque[float] = deque(maxlen=30)
        self._last_ts = time.monotonic()
        self._frame_count = 0

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detector_result,
        decision,
        temporal_result,
        ultrasonic_cm: float = -1.0,
        ssv2_sentence: str = "",
        depth=None,
        goal=None,
    ) -> np.ndarray:
        """Annotate a BGR frame with full AI pipeline state.

        goal: optional (x_norm, y_norm) in [0,1] — the user-selected navigation
        goal point (Phase 1: drawn only, does not yet drive motion).
        """
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        now = time.monotonic()
        self._fps_buf.append(1.0 / max(now - self._last_ts, 1e-6))
        self._last_ts = now
        self._frame_count += 1
        fps = float(np.mean(self._fps_buf))

        if self._overlay_det:
            self._draw_boxes(vis, detector_result)

        if self._overlay_risk:
            self._draw_risk_bar(vis, decision.risk_score, w, h)

        if self._overlay_action:
            # .value → "FORWARD" not "Action.FORWARD" (str-Enum on Python 3.11+),
            # so ACTION_COLOURS lookups hit and the HUD shows the right label/colour.
            self._draw_action(vis, getattr(decision.action, "value", decision.action), w, h)

        if self._overlay_wm:
            self._draw_wm_label(vis, getattr(decision.world_model_label, "value", decision.world_model_label))
            self._draw_temporal(vis, getattr(temporal_result.pattern, "value", temporal_result.pattern))

        if self._overlay_sonic:
            self._draw_sonic(vis, ultrasonic_cm, w)
        if self._overlay_mode_badge:
            self._draw_mode_badge(vis, w)
        if self._overlay_fps:
            self._draw_fps(vis, fps)
        if depth is not None and getattr(depth, "buffer_ready", False):
            # Spatial L/C/R free-space bars stay on the video; the text line is
            # optional (shown in the UI panel below by default).
            self._draw_depth(vis, depth, w, h, with_text=self._overlay_depth_text)
        if ssv2_sentence and self._overlay_ssv2:
            self._draw_ssv2(vis, ssv2_sentence, w, h)

        if goal is not None and getattr(goal, "active", False):
            self._draw_goal(vis, goal, w, h)

        return vis

    def _draw_goal(self, vis, goal, w: int, h: int) -> None:
        """Draw the tracked navigation goal: marker + heading arrow + readout.

        `goal` is a GoalState (active/lost/x/y/bearing_deg/distance_m). Phase 2 is
        display only — this does not steer the robot.
        """
        gx = min(max(float(getattr(goal, "x", 0.5)), 0.0), 1.0)
        gy = min(max(float(getattr(goal, "y", 0.5)), 0.0), 1.0)
        lost = bool(getattr(goal, "lost", False))
        reached = bool(getattr(goal, "reached", False))
        px, py = int(gx * w), int(gy * h)
        # green if reached, red if lost, else amber (BGR)
        colour = (0, 200, 0) if reached else ((0, 0, 255) if lost else (0, 215, 255))

        # Heading arrow from the image centre toward the goal (visualises bearing).
        cxp, cyp = w // 2, h // 2
        if not lost:
            cv2.arrowedLine(vis, (cxp, cyp), (px, py), colour, 2, cv2.LINE_AA, tipLength=0.15)

        # Crosshair + ring marker so it reads over any background.
        cv2.circle(vis, (px, py), 10, colour, 2, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 2, colour, -1, cv2.LINE_AA)
        for dx0, dy0, dx1, dy1 in ((-16, 0, -4, 0), (4, 0, 16, 0), (0, -16, 0, -4), (0, 4, 0, 16)):
            cv2.line(vis, (px + dx0, py + dy0), (px + dx1, py + dy1), colour, 2, cv2.LINE_AA)

        # Readout: REACHED, LOST, or bearing (deg, L/R) + distance.
        if reached:
            label = "GOAL REACHED"
        elif lost:
            label = "GOAL: lost"
        else:
            deg = float(getattr(goal, "bearing_deg", 0.0))
            side = "C" if abs(deg) < 1.0 else ("R" if deg > 0 else "L")
            dist = getattr(goal, "distance_m", None)
            dtxt = f"{dist:.1f}m" if dist else "?m"
            label = f"GOAL: {abs(deg):.0f}deg {side}  ~{dtxt}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = py - 16 if py - 16 - th > 0 else py + 28
        lx = int(min(max(px - tw // 2, 2), w - tw - 2))
        cv2.rectangle(vis, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (0, 0, 0), -1)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    def _draw_depth(self, vis, depth, w: int, h: int, with_text: bool = True) -> None:
        """Depth free-space HUD: L/C/R region bars (always) + optional distance text."""
        regions = getattr(depth, "region_distances_m", {}) or {}
        direction = getattr(depth, "clear_direction", "CENTER")
        ahead = getattr(depth, "clear_distance_m", -1.0)

        y = 78
        if with_text:
            # Text line near the top-left, under the WM/motion labels.
            text = f"Depth: {ahead:.2f}m ahead  |  open: {direction}"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(vis, (6, y - th - 5), (6 + tw + 6, y + 5), (0, 0, 0), -1)
            cv2.putText(vis, text, (9, y), font, scale, (0, 220, 255), thick, cv2.LINE_AA)

        # Three small bars (LEFT/CENTER/RIGHT), longer = more open; the chosen
        # direction is highlighted green.
        bx, by, bw, gap = 9, y + 10, 46, 6
        max_m = max([1.0] + [v for v in regions.values()])
        for i, name in enumerate(("LEFT", "CENTER", "RIGHT")):
            d = float(regions.get(name, 0.0))
            fill = int(bw * min(d / max_m, 1.0))
            x0 = bx + i * (bw + gap)
            colour = (0, 200, 0) if name == direction else (180, 180, 180)
            cv2.rectangle(vis, (x0, by), (x0 + bw, by + 8), (50, 50, 50), -1)
            cv2.rectangle(vis, (x0, by), (x0 + fill, by + 8), colour, -1)
            cv2.putText(vis, name[0], (x0 + bw // 2 - 3, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)

    def _draw_ssv2(self, vis, sentence: str, w: int, h: int) -> None:
        """Draw the genuine SSv2 action sentence (YOLO-filled) near the bottom."""
        text = f"SSv2: {sentence}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x, y = 10, h - 12
        cv2.rectangle(vis, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(vis, text, (x, y), font, scale, (0, 255, 255), thick, cv2.LINE_AA)

    def show(self, frame_bgr: np.ndarray) -> bool:
        """Show frame in OpenCV window. Returns False if user presses 'q'."""
        if not self._show:
            return True
        cv2.imshow(self._win, frame_bgr)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def encode_jpeg(self, frame_bgr: np.ndarray, quality: int = 80) -> bytes:
        """Encode annotated frame as JPEG bytes for TCP streaming."""
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def close(self) -> None:
        if self._show:
            cv2.destroyAllWindows()

    # ── Private ───────────────────────────────────────────────────────────────

    def _draw_boxes(self, vis: np.ndarray, det) -> None:
        for (x1, y1, x2, y2), label, conf in zip(
            det.boxes, det.labels, det.confidences
        ):
            c = CLASS_COLOURS.get(label, DEFAULT_COLOUR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), c, -1)
            cv2.putText(vis, txt, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_risk_bar(self, vis: np.ndarray, risk: float, w: int, h: int) -> None:
        bx, by = 8, h - 36
        bw, bh = w - 16, 18
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        fw = int(bw * risk)
        if fw > 0:
            r = int(min(risk * 2, 1.0) * 255)
            g = int(min((1 - risk) * 2, 1.0) * 255)
            cv2.rectangle(vis, (bx, by), (bx + fw, by + bh), (0, g, r), -1)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(vis, f"Risk: {risk:.2f}", (bx + 4, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    def _draw_action(self, vis: np.ndarray, action: str, w: int, h: int) -> None:
        c = ACTION_COLOURS.get(action, (200, 200, 200))
        txt = f"ACTION: {action}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = (w - tw) // 2, h - 55
        cv2.putText(vis, txt, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)

    def _draw_wm_label(self, vis: np.ndarray, label: str) -> None:
        c = WM_COLOURS.get(label, (150, 150, 150))
        txt = f"V-JEPA2: {label}"
        cv2.putText(vis, txt, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    def _draw_temporal(self, vis: np.ndarray, pattern: str) -> None:
        cv2.putText(vis, f"Motion: {pattern}", (8, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, f"Motion: {pattern}", (8, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    def _draw_sonic(self, vis: np.ndarray, cm: float, w: int) -> None:
        if cm < 0:
            return
        txt = f"Sonic: {cm:.1f}cm"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(vis, txt, (w - tw - 8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 50), 1)

    def _draw_mode_badge(self, vis: np.ndarray, w: int) -> None:
        txt = self._mode.upper()
        c = (0, 200, 100) if self._mode == "predictive" else (200, 130, 0)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = w - tw - 8
        cv2.rectangle(vis, (x - 4, 6), (x + tw + 4, 6 + th + 8), c, -1)
        cv2.putText(vis, txt, (x, 6 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_fps(self, vis: np.ndarray, fps: float) -> None:
        cv2.putText(vis, f"FPS:{fps:.1f} #{self._frame_count}",
                    (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
