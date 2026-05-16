"""
visualization.py – Real-time OpenCV overlay for the navigation HUD.

Draws on a copy of the current frame:
  • Bounding boxes for detected obstacles (colour-coded by class)
  • Risk bar: a horizontal gradient bar showing fused risk in [0,1]
  • Action label: large text showing the current navigation action
  • V-JEPA 2 prediction label (world model output)
  • Temporal motion pattern label
  • Navigation mode badge (PREDICTIVE / BASELINE)
  • Frame counter and FPS
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

# Colour palette for obstacle classes (BGR)
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "person":       (0, 165, 255),   # orange
    "chair":        (255, 0, 128),   # pink
    "couch":        (128, 0, 255),   # purple
    "dining table": (255, 128, 0),   # light blue
    "potted plant": (0, 200, 100),   # green
    "suitcase":     (200, 200, 0),   # teal
    "backpack":     (0, 200, 200),   # yellow
    "box":          (100, 100, 255), # red-ish
}
DEFAULT_COLOUR = (180, 180, 180)

ACTION_COLOURS = {
    "FORWARD":  (0, 220, 0),     # green
    "SLOW":     (0, 200, 255),   # yellow
    "STOP":     (0, 0, 220),     # red
    "REROUTE":  (0, 100, 255),   # orange-red
}

WM_LABEL_COLOURS = {
    "BLOCKED": (0, 0, 200),
    "CLEAR":   (0, 200, 0),
    "MIXED":   (0, 180, 255),
    "UNKNOWN": (150, 150, 150),
}


class Visualizer:
    def __init__(self, cfg: dict, navigation_mode: str = "predictive"):
        vis_cfg = cfg["visualization"]
        self._show = vis_cfg["show_window"]
        self._win_name = vis_cfg["window_name"]
        self._overlay_det = vis_cfg["overlay_detections"]
        self._overlay_risk = vis_cfg["overlay_risk_bar"]
        self._overlay_action = vis_cfg["overlay_action"]
        self._overlay_wm = vis_cfg["overlay_world_model_label"]
        self._nav_mode = navigation_mode

        self._fps_buf: deque[float] = deque(maxlen=30)
        self._last_ts = time.monotonic()
        self._frame_count = 0

    def annotate(
        self,
        frame: np.ndarray,
        detector_result,
        decision_result,
        temporal_result,
    ) -> np.ndarray:
        """Return a new annotated frame (does not modify the input)."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        now = time.monotonic()
        self._fps_buf.append(1.0 / max(now - self._last_ts, 1e-6))
        self._last_ts = now
        self._frame_count += 1
        fps = float(np.mean(self._fps_buf))

        if self._overlay_det:
            self._draw_detections(vis, detector_result)

        if self._overlay_risk:
            self._draw_risk_bar(vis, decision_result.risk_score, w, h)

        if self._overlay_action:
            self._draw_action(vis, decision_result.action, w, h)

        if self._overlay_wm:
            self._draw_wm_label(vis, decision_result.world_model_label, w)

        self._draw_temporal_label(vis, temporal_result.pattern, w)
        self._draw_mode_badge(vis, w)
        self._draw_fps(vis, fps, decision_result.risk_score)

        return vis

    def show(self, frame: np.ndarray) -> bool:
        """Display the frame. Returns False if the user pressed 'q'."""
        if not self._show:
            return True
        cv2.imshow(self._win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        if self._show:
            cv2.destroyAllWindows()

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_detections(self, vis: np.ndarray, det) -> None:
        for (x1, y1, x2, y2), label, conf in zip(det.boxes, det.labels, det.confidences):
            colour = CLASS_COLOURS.get(label, DEFAULT_COLOUR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_risk_bar(self, vis: np.ndarray, risk: float, w: int, h: int) -> None:
        bar_x, bar_y = 10, h - 40
        bar_w, bar_h = w - 20, 20
        # Background
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Filled portion – green→yellow→red gradient
        filled_w = int(bar_w * risk)
        if filled_w > 0:
            r = int(min(risk * 2, 1.0) * 255)
            g = int(min((1 - risk) * 2, 1.0) * 255)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, g, r), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        cv2.putText(vis, f"Risk: {risk:.2f}", (bar_x + 4, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_action(self, vis: np.ndarray, action: str, w: int, h: int) -> None:
        colour = ACTION_COLOURS.get(action, (200, 200, 200))
        text = f"ACTION: {action}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (w - tw) // 2
        y = h - 70
        # Shadow
        cv2.putText(vis, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    def _draw_wm_label(self, vis: np.ndarray, label: str, w: int) -> None:
        colour = WM_LABEL_COLOURS.get(label, (150, 150, 150))
        text = f"V-JEPA2: {label}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    def _draw_temporal_label(self, vis: np.ndarray, pattern: str, w: int) -> None:
        text = f"Motion: {pattern}"
        cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    def _draw_mode_badge(self, vis: np.ndarray, w: int) -> None:
        mode_text = self._nav_mode.upper()
        colour = (0, 200, 100) if self._nav_mode == "predictive" else (200, 130, 0)
        (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = w - tw - 10
        cv2.rectangle(vis, (x - 4, 10), (x + tw + 4, 10 + th + 8), colour, -1)
        cv2.putText(vis, mode_text, (x, 10 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

    def _draw_fps(self, vis: np.ndarray, fps: float, risk: float) -> None:
        text = f"FPS: {fps:.1f}  Frame: {self._frame_count}"
        cv2.putText(vis, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1)
