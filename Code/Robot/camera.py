"""
camera.py – Pi camera wrapper (Freenove FNK0077 streaming approach).

Primary backend:  picamera2 with the hardware JpegEncoder driving a
                  StreamingOutput buffer via start_recording() — this is the
                  exact proven-working pattern from Freenove's Code/Server/camera.py.
Fallback backend: cv2.VideoCapture on /dev/video<device> for Docker/dev
                  environments where libcamera is not installed.  A background
                  thread grabs + JPEG-encodes frames into the same
                  StreamingOutput so get_frame() behaves identically.

Image orientation:
  The Freenove FNK0077 mounts the CSI camera upside-down, so hflip and vflip
  both default to True (matching the Freenove original) to produce a
  correctly-oriented image.  Set both False if your mount is upright.
"""
from __future__ import annotations

import io
import logging
import threading
from threading import Condition

logger = logging.getLogger(__name__)


class StreamingOutput(io.BufferedIOBase):
    """Thread-safe latest-JPEG-frame buffer (Freenove pattern)."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf) -> None:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


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

        self._picam = None       # picamera2 backend
        self._cap = None         # OpenCV VideoCapture fallback
        self._streaming = False

        self.streaming_output = StreamingOutput()

        # OpenCV fallback capture thread
        self._fallback_thread: threading.Thread | None = None
        self._running = False

    def start_stream(self) -> None:
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import JpegEncoder
            from picamera2.outputs import FileOutput
            from libcamera import Transform

            transform = Transform(
                hflip=1 if self._hflip else 0,
                vflip=1 if self._vflip else 0,
            )
            self._picam = Picamera2()
            stream_config = self._picam.create_video_configuration(
                main={"size": (self._width, self._height)},
                transform=transform,
            )
            self._picam.configure(stream_config)
            # Hardware JPEG encoder pushes frames into streaming_output.write()
            self._picam.start_recording(
                JpegEncoder(), FileOutput(self.streaming_output)
            )
            self._streaming = True
            logger.info(
                "Camera streaming via picamera2 JpegEncoder at %dx%d (hflip=%s vflip=%s)",
                self._width, self._height, self._hflip, self._vflip,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "picamera2/libcamera unavailable (%s); "
                "falling back to OpenCV V4L2 (/dev/video%d)",
                exc, self._device,
            )
            self._start_opencv_fallback()

    def _start_opencv_fallback(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self._device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open /dev/video{self._device} – "
                "check '--device /dev/video0:/dev/video0' in docker run"
            )
        logger.info(
            "Camera streaming via OpenCV V4L2 (/dev/video%d) at %dx%d (hflip=%s vflip=%s)",
            self._device, self._width, self._height, self._hflip, self._vflip,
        )
        self._running = True
        self._fallback_thread = threading.Thread(
            target=self._opencv_capture_loop, daemon=True, name="CameraCapture"
        )
        self._fallback_thread.start()

    def _opencv_capture_loop(self) -> None:
        """Grab + JPEG-encode frames into streaming_output (fallback only)."""
        import cv2
        while self._running:
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                continue
            # picamera2 applies flips via Transform; do the same here for parity
            if self._hflip and self._vflip:
                frame_bgr = cv2.flip(frame_bgr, -1)   # 180° rotation
            elif self._hflip:
                frame_bgr = cv2.flip(frame_bgr, 1)    # horizontal
            elif self._vflip:
                frame_bgr = cv2.flip(frame_bgr, 0)    # vertical
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if ok:
                self.streaming_output.write(buf.tobytes())

    def get_frame(self) -> bytes | None:
        """Block until a new JPEG frame is available, then return it (Freenove pattern)."""
        with self.streaming_output.condition:
            # Timeout so shutdown / disconnect does not block the thread forever
            self.streaming_output.condition.wait(timeout=1.0)
            return self.streaming_output.frame

    def close(self) -> None:
        self._running = False
        # Wake any thread blocked in get_frame()
        with self.streaming_output.condition:
            self.streaming_output.condition.notify_all()
        if self._picam:
            try:
                if self._streaming:
                    self._picam.stop_recording()
                self._picam.close()
            except Exception as exc:
                logger.warning("Camera close error: %s", exc)
            finally:
                self._picam = None
                self._streaming = False
        if self._cap:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Camera release error: %s", exc)
            finally:
                self._cap = None
        logger.info("Camera closed")
