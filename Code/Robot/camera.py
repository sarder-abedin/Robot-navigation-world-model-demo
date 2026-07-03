"""
camera.py – Pi camera wrapper with OpenCV V4L2 fallback.

Primary backend:  picamera2 (requires system python3-libcamera via apt).
Fallback backend: cv2.VideoCapture on /dev/video<device> (works in Docker
                  when --device /dev/video0:/dev/video0 is passed).

The fallback activates automatically when libcamera is not installed,
so the Docker image works without the Raspberry Pi Foundation apt repo.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, stream_size: tuple[int, int] = (400, 300), device: int = 0):
        self._width, self._height = stream_size
        self._device = device
        self._picam = None   # picamera2 backend
        self._cap = None     # OpenCV VideoCapture fallback
        self._lock = threading.Lock()

    def start_stream(self) -> None:
        try:
            from picamera2 import Picamera2
            self._picam = Picamera2()
            config = self._picam.create_video_configuration(
                main={"format": "RGB888", "size": (self._width, self._height)}
            )
            self._picam.configure(config)
            self._picam.start()
            logger.info("Camera started via picamera2 at %dx%d", self._width, self._height)
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "picamera2/libcamera unavailable (%s); "
                "falling back to OpenCV V4L2 (/dev/video%d)",
                exc, self._device,
            )
            import cv2
            self._cap = cv2.VideoCapture(self._device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to open /dev/video{self._device} – "
                    "check '--device /dev/video0:/dev/video0' in docker run"
                ) from exc
            logger.info(
                "Camera started via OpenCV V4L2 (/dev/video%d) at %dx%d",
                self._device, self._width, self._height,
            )

    def get_frame(self) -> bytes | None:
        import cv2
        if self._picam is not None:
            with self._lock:
                frame_rgb = self._picam.capture_array()
            if frame_rgb is None:
                return None
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif self._cap is not None:
            with self._lock:
                ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                return None
        else:
            return None
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
        if self._cap:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Camera release error: %s", exc)
            finally:
                self._cap = None
        logger.info("Camera closed")
