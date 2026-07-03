"""
camera.py – Pi camera wrapper with OpenCV V4L2 fallback.

Primary backend:  picamera2 (requires system python3-libcamera via apt).
Fallback backend: cv2.VideoCapture on /dev/video<device> (works in Docker
                  when --device /dev/video0:/dev/video0 is passed).

The fallback activates automatically when libcamera is not installed.

A background capture thread continuously buffers the latest JPEG frame so
that both the streaming thread and the detection thread read the same frame
without issuing duplicate capture calls.

Image orientation:
  The Freenove FNK0077 mounts the CSI camera upside-down.  Pass
  hflip=True, vflip=True (the defaults) to rotate 180° and get a
  correctly-oriented image.  Set both to False if your mount is upright.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class Camera:
    def __init__(
        self,
        stream_size: tuple[int, int] = (400, 300),
        device: int = 0,
        hflip: bool = True,
        vflip: bool = True,
    ):
        self._width, self._height = stream_size
        self._device = device
        self._hflip = hflip
        self._vflip = vflip

        self._picam = None   # picamera2 backend
        self._cap = None     # OpenCV VideoCapture fallback

        # Shared frame buffer – written by background thread, read by callers
        self._frame_lock = threading.Lock()
        self._latest_jpg: bytes | None = None
        self._frame_event = threading.Event()

        self._capture_thread: threading.Thread | None = None
        self._running = False

    def start_stream(self) -> None:
        self._running = True
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
            transform = Transform(
                hflip=1 if self._hflip else 0,
                vflip=1 if self._vflip else 0,
            )
            self._picam = Picamera2()
            config = self._picam.create_video_configuration(
                main={"format": "RGB888", "size": (self._width, self._height)},
                transform=transform,
            )
            self._picam.configure(config)
            self._picam.start()
            logger.info(
                "Camera started via picamera2 at %dx%d (hflip=%s vflip=%s)",
                self._width, self._height, self._hflip, self._vflip,
            )
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
                "Camera started via OpenCV V4L2 (/dev/video%d) at %dx%d (hflip=%s vflip=%s)",
                self._device, self._width, self._height, self._hflip, self._vflip,
            )

        # Start background capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Continuously capture frames into the shared buffer."""
        import cv2
        while self._running:
            try:
                frame_bgr = self._grab_bgr()
                if frame_bgr is None:
                    continue
                ok, buf = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ok:
                    jpg = buf.tobytes()
                    with self._frame_lock:
                        self._latest_jpg = jpg
                    self._frame_event.set()
            except Exception as exc:
                logger.warning("Camera capture error: %s", exc)

    def _grab_bgr(self):
        """Grab one BGR frame from whichever backend is active."""
        import cv2
        if self._picam is not None:
            frame_rgb = self._picam.capture_array()
            if frame_rgb is None:
                return None
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif self._cap is not None:
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                return None
            # Apply flip for OpenCV fallback (picamera2 uses Transform in config)
            if self._hflip and self._vflip:
                frame_bgr = cv2.flip(frame_bgr, -1)   # 180° rotation
            elif self._hflip:
                frame_bgr = cv2.flip(frame_bgr, 1)    # horizontal
            elif self._vflip:
                frame_bgr = cv2.flip(frame_bgr, 0)    # vertical
            return frame_bgr
        return None

    def get_frame(self) -> bytes | None:
        """Return the latest JPEG frame from the shared buffer."""
        with self._frame_lock:
            return self._latest_jpg

    def close(self) -> None:
        self._running = False
        self._frame_event.set()  # unblock any waiting thread
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
