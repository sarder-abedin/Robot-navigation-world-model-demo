"""
camera.py – Pi camera wrapper (continuous JPEG-encoder streaming).

picamera2's on-demand capture_array() is flaky inside Docker (the container
often can't open /dev/dma_heap/vidbuf_cached, so it uses a limited buffer pool
and capture_array() intermittently blocks after the first frame). To stream
reliably we use the hardware JpegEncoder + a StreamingOutput buffer via
start_recording() — the same continuous-streaming pattern Freenove uses, which
manages buffers internally and never starves on a per-call basis.

Primary backend:  picamera2 JpegEncoder → StreamingOutput (continuous).
Fallback backend: cv2.VideoCapture on /dev/video<device>, with a background
                  grab-and-encode thread feeding the same StreamingOutput.

Image orientation:
  Applied at capture (Transform for picamera2, cv2.flip for OpenCV), so the
  corrected frame is what streams to the PC (UI, V-JEPA 2, YOLO). Default is
  upright (no flip); set hflip/vflip True for an inverted mount.
"""
from __future__ import annotations

import io
import logging
import threading
from threading import Condition

logger = logging.getLogger(__name__)


class StreamingOutput(io.BufferedIOBase):
    """Thread-safe holder for the most recent JPEG frame (Freenove pattern)."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf) -> None:
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()


class Camera:
    def __init__(
        self,
        stream_size: tuple[int, int] = (400, 300),
        device: int = 0,
        hflip: bool = False,
        vflip: bool = False,
    ):
        self._width, self._height = stream_size
        self._device = device
        self._hflip = hflip
        self._vflip = vflip
        self._picam = None            # picamera2 backend
        self._cap = None              # OpenCV VideoCapture fallback
        self._output = StreamingOutput()
        self._lock = threading.Lock()
        self._running = False
        self._fallback_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_stream(self) -> None:
        try:
            from libcamera import Transform
            from picamera2 import Picamera2
            from picamera2.encoders import JpegEncoder
            from picamera2.outputs import FileOutput

            # libcamera enumerates cameras via the udev socket; in Docker that
            # must be mounted or global_camera_info() is empty and Picamera2()
            # raises an opaque IndexError. Check first with a clear message.
            cameras = Picamera2.global_camera_info()
            if not cameras:
                raise RuntimeError(
                    "libcamera found no cameras. In Docker this almost always means "
                    "the udev socket is not mounted — add '-v /run/udev:/run/udev:ro' "
                    "to your docker run (or use docker-compose.robot.yml, which mounts it). "
                    "Confirm the CSI ribbon and: rpicam-hello --list-cameras"
                )
            transform = Transform(
                hflip=1 if self._hflip else 0,
                vflip=1 if self._vflip else 0,
            )
            self._picam = Picamera2()
            config = self._picam.create_video_configuration(
                main={"size": (self._width, self._height)},
                transform=transform,
            )
            self._picam.configure(config)
            # Hardware JPEG encoder pushes frames continuously into _output.write().
            self._picam.start_recording(JpegEncoder(), FileOutput(self._output))
            self._running = True
            logger.info(
                "Camera streaming via picamera2 JpegEncoder at %dx%d (hflip=%s vflip=%s)",
                self._width, self._height, self._hflip, self._vflip,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning(
                "picamera2/libcamera unavailable (%s); falling back to OpenCV V4L2 (/dev/video%d)",
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
        self._running = True
        self._fallback_thread = threading.Thread(
            target=self._opencv_capture_loop, daemon=True, name="CameraCapture"
        )
        self._fallback_thread.start()
        logger.info(
            "Camera streaming via OpenCV V4L2 (/dev/video%d) at %dx%d (hflip=%s vflip=%s)",
            self._device, self._width, self._height, self._hflip, self._vflip,
        )

    def _opencv_capture_loop(self) -> None:
        import cv2
        while self._running and self._cap is not None:
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                continue
            frame_bgr = self._apply_flip(frame_bgr)
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self._output.write(buf.tobytes())

    def _apply_flip(self, frame_bgr):
        import cv2
        if self._hflip and self._vflip:
            return cv2.flip(frame_bgr, -1)   # 180°
        if self._hflip:
            return cv2.flip(frame_bgr, 1)
        if self._vflip:
            return cv2.flip(frame_bgr, 0)
        return frame_bgr

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frame(self) -> bytes | None:
        """Block until a new JPEG frame is available, then return it.

        Reads the continuously-filled StreamingOutput — no per-call capture, so
        it can't starve the picamera2 buffer pool. The 1 s timeout means it
        returns the latest frame (or None at startup) rather than blocking
        forever if the encoder stalls.
        """
        with self._output.condition:
            self._output.condition.wait(timeout=1.0)
            return self._output.frame

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._running = False
        # Wake anything blocked in get_frame().
        with self._output.condition:
            self._output.condition.notify_all()
        with self._lock:
            picam, self._picam = self._picam, None
            cap, self._cap = self._cap, None
        if picam:
            try:
                picam.stop_recording()
                picam.close()
            except Exception as exc:
                logger.warning("Camera close error: %s", exc)
        if cap:
            try:
                cap.release()
            except Exception as exc:
                logger.warning("Camera release error: %s", exc)
        logger.info("Camera closed")
