"""
camera.py – picamera2 wrapper for the Pi robot client.

Captures frames from the Pi camera, JPEG-encodes them, and exposes a
simple get_frame() interface so main_robot.py can stream to the PC.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, stream_size: tuple[int, int] = (400, 300)):
        self._width, self._height = stream_size
        self._picam = None
        self._lock = threading.Lock()

    def start_stream(self) -> None:
        from picamera2 import Picamera2
        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"format": "RGB888", "size": (self._width, self._height)}
        )
        self._picam.configure(config)
        self._picam.start()
        logger.info("Camera started at %dx%d", self._width, self._height)

    def get_frame(self) -> bytes | None:
        if self._picam is None:
            return None
        with self._lock:
            frame_rgb = self._picam.capture_array()
        if frame_rgb is None:
            return None
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def close(self) -> None:
        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as exc:
                logger.warning("Camera close error: %s", exc)
            finally:
                self._picam = None
        logger.info("Camera closed")
