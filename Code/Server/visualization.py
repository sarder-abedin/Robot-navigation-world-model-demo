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
        self._overlay_wm = vis.get("overlay_world_model_label", True)
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
    ) -> np.ndarray:
        """Annotate a BGR frame with full AI pipeline state."""
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
            self._draw_action(vis, str(decision.action), w, h)

        if self._overlay_wm:
            self._draw_wm_label(vis, decision.world_model_label)
            self._draw_temporal(vis, temporal_result.pattern)

        self._draw_sonic(vis, ultrasonic_cm, w)
        self._draw_mode_badge(vis, w)
        self._draw_fps(vis, fps)

        return vis

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
