"""
camera.py – Pi camera wrapper (continuous JPEG-encoder streaming + stall recovery).

picamera2's on-demand capture_array() is flaky inside Docker (the container
often can't open /dev/dma_heap/vidbuf_cached, so it uses a limited buffer pool
and capture_array() intermittently blocks after the first frame). To stream
reliably we use the hardware JpegEncoder + a StreamingOutput buffer via
start_recording() — the same continuous-streaming pattern Freenove uses, which
manages buffers internally and never starves on a per-call basis.

Even so, on the limited buffer pool the encoder can eventually **stall** (stops
delivering frames while the socket stays up), which would silently freeze the
stream — the PC then keeps receiving the same frame and watchdog-STOPs the robot.
So a background **watchdog** restarts the camera backend when no new frame has
arrived for `stall_timeout` seconds, and get_frame() reports *no frame* (rather
than the frozen one) during a stall so the robot never drives on a stale image.
Mounting `-v /dev/dma_heap:/dev/dma_heap` avoids the stall in the first place;
the watchdog is the safety net for when it still happens.

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
import time
from threading import Condition

logger = logging.getLogger(__name__)


def _is_stale(last_ts, now: float, timeout: float) -> bool:
    """True if the last frame is older than `timeout` (None ts → never written)."""
    return last_ts is not None and (now - last_ts) > timeout


def _should_restart(seconds_idle, timeout: float) -> bool:
    """Watchdog decision: restart when idle for longer than `timeout`.

    seconds_idle is StreamingOutput.seconds_idle() — None means no frame has
    arrived yet, which (after reset_timer at start) still counts up, so a camera
    that starts but never delivers is recovered too."""
    return seconds_idle is not None and seconds_idle > timeout


class StreamingOutput(io.BufferedIOBase):
    """Thread-safe holder for the most recent JPEG frame (Freenove pattern),
    plus a monotonic timestamp of the last write so a stall is detectable."""

    def __init__(self, clock=time.monotonic) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()
        self.seq = 0                       # frames written (monotonic counter)
        self.last_write_ts: float | None = None   # monotonic ts of last write
        self._clock = clock

    def write(self, buf) -> None:
        with self.condition:
            self.frame = bytes(buf)
            self.seq += 1
            self.last_write_ts = self._clock()
            self.condition.notify_all()

    def reset_timer(self) -> None:
        """Mark 'alive now' so a freshly (re)started backend gets a grace window
        before the stall watchdog can fire again."""
        with self.condition:
            self.last_write_ts = self._clock()

    def seconds_idle(self):
        """Seconds since the last write, or None if nothing has been written."""
        with self.condition:
            if self.last_write_ts is None:
                return None
            return self._clock() - self.last_write_ts


class Camera:
    def __init__(
        self,
        stream_size: tuple[int, int] = (400, 300),
        device: int = 0,
        hflip: bool = False,
        vflip: bool = False,
        stall_timeout_s: float = 2.5,
        watchdog_interval_s: float = 1.0,
        clock=time.monotonic,
    ):
        self._width, self._height = stream_size
        self._device = device
        self._hflip = hflip
        self._vflip = vflip
        self._stall_timeout = max(0.5, float(stall_timeout_s))
        self._watchdog_interval = max(0.2, float(watchdog_interval_s))
        self._clock = clock
        self._picam = None            # picamera2 backend
        self._cap = None              # OpenCV VideoCapture fallback
        self._output = StreamingOutput(clock=clock)
        self._lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._running = False
        self._restarts = 0
        self._fallback_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_stream(self) -> None:
        self._running = True
        try:
            self._start_backend()
        except Exception:
            self._running = False
            raise
        self._output.reset_timer()        # grace window before the watchdog can fire
        self._start_watchdog()

    def _start_backend(self) -> None:
        """Bring up the picamera2 backend, or fall back to OpenCV V4L2. Raises on
        a hard failure (no camera at all). Reused by start_stream() and restart()."""
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
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (self._width, self._height)},
                transform=transform,
            )
            picam.configure(config)
            # Hardware JPEG encoder pushes frames continuously into _output.write().
            picam.start_recording(JpegEncoder(), FileOutput(self._output))
            self._picam = picam
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
        cap = cv2.VideoCapture(self._device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to open /dev/video{self._device} – "
                "check '--device /dev/video0:/dev/video0' in docker run"
            )
        self._cap = cap
        self._fallback_thread = threading.Thread(
            target=self._opencv_capture_loop, args=(cap,), daemon=True, name="CameraCapture"
        )
        self._fallback_thread.start()
        logger.info(
            "Camera streaming via OpenCV V4L2 (/dev/video%d) at %dx%d (hflip=%s vflip=%s)",
            self._device, self._width, self._height, self._hflip, self._vflip,
        )

    def _opencv_capture_loop(self, cap) -> None:
        # Bind to the capture this thread was started with; a restart swaps
        # self._cap, so `self._cap is cap` cleanly stops the old loop.
        import cv2
        while self._running and self._cap is cap:
            ok, frame_bgr = cap.read()
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

    # ── Stall watchdog (auto-recovery) ─────────────────────────────────────────

    def _start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="CameraWatchdog"
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while self._running:
            time.sleep(self._watchdog_interval)
            if not self._running:
                break
            idle = self._output.seconds_idle()
            if _should_restart(idle, self._stall_timeout):
                logger.warning(
                    "Camera stalled (%.1fs since last frame) – restarting the backend",
                    idle,
                )
                self._restart()

    def _restart(self) -> None:
        with self._restart_lock:
            if not self._running:
                return
            self._stop_backend()
            time.sleep(0.4)                 # let the device release before re-acquiring
            try:
                self._start_backend()
                self._restarts += 1
                logger.info("Camera backend restarted (restart #%d)", self._restarts)
            except Exception as exc:
                logger.error("Camera restart failed (%s) – retrying next tick", exc)
            finally:
                # Either way, reset the timer so the watchdog waits a full
                # stall_timeout before firing again (no restart storm).
                self._output.reset_timer()

    def _stop_backend(self) -> None:
        """Tear down the current backend (picamera2 or OpenCV) without touching
        _running/the watchdog, so a restart can bring a fresh one straight back up."""
        with self._lock:
            picam, self._picam = self._picam, None
            cap, self._cap = self._cap, None
            thread, self._fallback_thread = self._fallback_thread, None
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)        # the loop exits once self._cap is not cap
        if picam:
            try:
                picam.stop_recording()
                picam.close()
            except Exception as exc:
                logger.warning("Camera stop error: %s", exc)
        if cap:
            try:
                cap.release()
            except Exception as exc:
                logger.warning("Camera release error: %s", exc)

    # ── Frame access ──────────────────────────────────────────────────────────

    def _frame_or_none(self, frame, last_ts):
        """Return the frame unless it's stale (encoder stalled) — in which case
        report None so the robot doesn't stream/drive on a frozen image while the
        watchdog restarts the backend."""
        if frame is None:
            return None
        if _is_stale(last_ts, self._clock(), self._stall_timeout):
            return None
        return frame

    def get_frame(self) -> bytes | None:
        """Block briefly for a new JPEG frame, then return it (or None).

        Reads the continuously-filled StreamingOutput — no per-call capture, so it
        can't starve the picamera2 buffer pool. Returns None at startup, or when
        the encoder has stalled (the frame is frozen), rather than handing back a
        stale image; the watchdog restarts the backend in that case.
        """
        with self._output.condition:
            self._output.condition.wait(timeout=1.0)
            frame, last_ts = self._output.frame, self._output.last_write_ts
        return self._frame_or_none(frame, last_ts)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._running = False
        # Wake anything blocked in get_frame().
        with self._output.condition:
            self._output.condition.notify_all()
        wd = self._watchdog_thread
        if wd and wd is not threading.current_thread():
            wd.join(timeout=2.0)
        self._stop_backend()
        logger.info("Camera closed")
