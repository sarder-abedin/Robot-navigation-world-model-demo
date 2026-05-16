"""
camera.py – Frame source abstraction.

Supports two backends:
  - LiveCamera   : reads from a physical camera via OpenCV (live robot mode)
  - DemoCamera   : reads from a recorded video file (offline / demo mode)

Both expose the same interface: open(), read() -> (ok, frame), release().
"""

import time
import cv2
import logging

logger = logging.getLogger(__name__)


class LiveCamera:
    """Reads frames from a physical camera attached to the robot."""

    def __init__(self, cfg: dict):
        self._index = cfg["device_index"]
        self._width = cfg["frame_width"]
        self._height = cfg["frame_height"]
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self._index}")
        logger.info("Live camera opened on device %d", self._index)

    def read(self):
        if self._cap is None:
            raise RuntimeError("Camera not opened – call open() first")
        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            logger.info("Live camera released")


class DemoCamera:
    """
    Reads frames from a recorded video file.

    When the video ends it loops back to the start so the demo can run
    indefinitely without extra command-line handling.
    """

    def __init__(self, cfg: dict):
        self._path = cfg["demo_video_path"]
        self._target_fps = cfg.get("fps", 30)
        self._cap: cv2.VideoCapture | None = None
        self._frame_delay = 1.0 / self._target_fps
        self._last_read = 0.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open demo video: {self._path}")
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self._target_fps
        self._frame_delay = 1.0 / actual_fps
        logger.info("Demo camera opened: %s (%.1f fps)", self._path, actual_fps)

    def read(self):
        if self._cap is None:
            raise RuntimeError("Camera not opened – call open() first")

        # Throttle reads to match the recorded frame-rate
        now = time.monotonic()
        elapsed = now - self._last_read
        if elapsed < self._frame_delay:
            time.sleep(self._frame_delay - elapsed)

        ok, frame = self._cap.read()
        if not ok:
            # End of file – loop back
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()

        self._last_read = time.monotonic()
        return ok, frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            logger.info("Demo camera released")


def build_camera(cfg: dict):
    """Factory: return the right camera backend based on config."""
    mode = cfg.get("mode", "demo")
    cam_cfg = cfg["camera"]
    if mode == "live":
        return LiveCamera(cam_cfg)
    return DemoCamera(cam_cfg)
