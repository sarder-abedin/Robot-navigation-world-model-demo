"""
camera.py – Pi camera wrapper (on-demand capture for split-inference).

In this split-inference architecture the Pi grabs one frame at a time and
sends it to the PC, so on-demand capture (picamera2 capture_array) is simpler
and more robust than a continuous JpegEncoder stream — every get_frame() call
is guaranteed to return the latest frame or None, never blocks indefinitely.

Primary backend:  picamera2 capture_array() with a Transform for hflip/vflip.
Fallback backend: cv2.VideoCapture on /dev/video<device> (works in Docker
                  when --device /dev/video0:/dev/video0 is passed).

Image orientation:
  Orientation is applied here at capture, so the corrected frame is what gets
  streamed to the PC (UI, V-JEPA 2, YOLO). Default is upright (no flip); set
  hflip/vflip True (config or -e CAMERA_HFLIP/CAMERA_VFLIP) for an inverted mount.
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
        hflip: bool = False,
        vflip: bool = False,
    ):
        self._width, self._height = stream_size
        self._device = device
        self._hflip = hflip
        self._vflip = vflip
        self._picam = None   # picamera2 backend
        self._cap = None     # OpenCV VideoCapture fallback
        self._lock = threading.Lock()

    def start_stream(self) -> None:
        try:
            from picamera2 import Picamera2
            from libcamera import Transform

            transform = Transform(
                hflip=1 if self._hflip else 0,
                vflip=1 if self._vflip else 0,
            )
            # libcamera enumerates cameras via the udev socket. In Docker that
            # socket must be mounted; otherwise global_camera_info() is empty and
            # Picamera2() raises an opaque IndexError. Check first and give a
            # clear, actionable message.
            cameras = Picamera2.global_camera_info()
            if not cameras:
                raise RuntimeError(
                    "libcamera found no cameras. In Docker this almost always means "
                    "the udev socket is not mounted — add '-v /run/udev:/run/udev:ro' "
                    "to your docker run (or use 'docker compose -f docker-compose.robot.yml up', "
                    "which already mounts it). Also confirm the CSI ribbon is seated and the "
                    "camera works on the host with: rpicam-hello --list-cameras"
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
                "Camera started via OpenCV V4L2 (/dev/video%d) at %dx%d",
                self._device, self._width, self._height,
            )

    def _apply_flip(self, frame_bgr):
        import cv2
        # picamera2 applies flips via Transform; the OpenCV fallback must do it here
        if self._hflip and self._vflip:
            return cv2.flip(frame_bgr, -1)   # 180° rotation
        if self._hflip:
            return cv2.flip(frame_bgr, 1)    # horizontal
        if self._vflip:
            return cv2.flip(frame_bgr, 0)    # vertical
        return frame_bgr

    def get_frame(self) -> bytes | None:
        """Capture one frame and return it as JPEG bytes (or None on failure)."""
        import cv2
        if self._picam is not None:
            with self._lock:
                # picamera2's "RGB888" format returns a BGR-ordered array (its
                # names are little-endian, so "RGB888" is B,G,R in memory — the
                # OpenCV-native order). Use it directly; converting RGB->BGR here
                # would swap red and blue and tint the whole stream red.
                frame_bgr = self._picam.capture_array()
            if frame_bgr is None:
                return None
        elif self._cap is not None:
            with self._lock:
                ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                return None
            frame_bgr = self._apply_flip(frame_bgr)
        else:
            return None
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def close(self) -> None:
        # Take the capture lock so close() can't run mid-capture_array()/read()
        # in the streaming thread (which would hang or crash picamera2 on stop).
        with self._lock:
            picam, self._picam = self._picam, None
            cap, self._cap = self._cap, None
        if picam:
            try:
                picam.stop()
                picam.close()
            except Exception as exc:
                logger.warning("Camera close error: %s", exc)
        if cap:
            try:
                cap.release()
            except Exception as exc:
                logger.warning("Camera release error: %s", exc)
        logger.info("Camera closed")
