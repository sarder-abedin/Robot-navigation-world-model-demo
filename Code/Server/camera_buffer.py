"""
camera_buffer.py – Rolling frame buffer that adapts the Freenove Pi Camera.

In live mode the Freenove camera streams JPEG bytes via picamera2's JpegEncoder.
We hook into the StreamingOutput to decode each new JPEG into a numpy RGB array
and keep a rolling deque so the AI pipeline always has a ready clip.

In demo mode we read from a video file (OpenCV) at the same frame rate so the
full pipeline can be exercised on a laptop without a Raspberry Pi camera.

Public API:
  buf = CameraBuffer(cfg)
  buf.start()                       # launches background capture thread
  frame = buf.get_latest_frame()    # most recent (H, W, 3) uint8 RGB array
  clip  = buf.get_clip(n=16)        # list of n most recent frames, or None
  buf.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraBuffer:
    """
    Frame buffer supporting three source modes:

    demo  – reads from a recorded video file (OpenCV)
    live  – reads from Freenove picamera2 StreamingOutput
    tcp   – frames are pushed externally via push_frame() (robot TCP stream)
    """

    def __init__(self, cfg: dict, freenove_camera=None):
        cam_cfg = cfg["camera"]
        self._clip_len = cam_cfg["clip_length"]
        self._ai_size = cam_cfg["ai_frame_size"]
        self._mode = cfg.get("mode", "demo")

        # Shared rolling deque; maxlen keeps memory bounded
        self._buf: deque[np.ndarray] = deque(maxlen=self._clip_len)
        self._lock = threading.Lock()
        # Monotonic counter bumped whenever a NEW frame is appended, so the AI
        # pipeline can skip re-processing the same frame thousands of times.
        self._seq = 0

        self._thread: threading.Thread | None = None
        self._running = False

        if self._mode == "live":
            if freenove_camera is None:
                raise ValueError("freenove_camera is required in live mode")
            self._stream_output = freenove_camera.streaming_output
            self._cap = None
        elif self._mode == "tcp":
            # Frames are pushed via push_frame(); no background thread needed
            self._stream_output = None
            self._cap = None
        else:
            self._stream_output = None
            self._video_path = cam_cfg["demo_video_path"]
            self._cap = None  # opened on start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        if self._mode == "live":
            self._thread = threading.Thread(
                target=self._live_loop, daemon=True, name="CameraBufferLive"
            )
            self._thread.start()
        elif self._mode == "tcp":
            # No capture thread – frames arrive via push_frame()
            pass
        else:
            self._cap = cv2.VideoCapture(self._video_path)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open demo video: {self._video_path}\n"
                    "  To run with a real Pi robot use LIVE mode:\n"
                    "    docker run -e NAV_MODE=live ... nav-server\n"
                    "    docker compose -f docker-compose.server.yml up  "
                    "(with NAV_MODE=live)\n"
                    "  To run demo mode without a robot, place a corridor video at:\n"
                    f"    {self._video_path}"
                )
            self._thread = threading.Thread(
                target=self._demo_loop, daemon=True, name="CameraBufferDemo"
            )
            self._thread.start()
        logger.info("CameraBuffer started in %s mode", self._mode)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("CameraBuffer stopped")

    # ── Public frame access ───────────────────────────────────────────────────

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def get_latest(self):
        """Return (seq, frame) of the most recent frame, or None if empty.

        `seq` lets the caller detect whether a genuinely new frame has arrived
        instead of reprocessing the same one at thousands of fps.
        """
        with self._lock:
            if not self._buf:
                return None
            return self._seq, self._buf[-1]

    def get_clip(self, n: int | None = None) -> list[np.ndarray] | None:
        """Return the last n frames as a list, or None if the buffer is not full."""
        n = n or self._clip_len
        with self._lock:
            frames = list(self._buf)
        if len(frames) < n:
            return None
        return frames[-n:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    # ── External push (TCP mode) ──────────────────────────────────────────────

    def push_frame(self, jpg: bytes) -> None:
        """Push a JPEG frame from an external source (TCP robot stream)."""
        frame = _decode_jpeg(jpg, self._ai_size)
        if frame is not None:
            with self._lock:
                self._buf.append(frame)
                self._seq += 1

    # ── Background capture loops ──────────────────────────────────────────────

    def _live_loop(self) -> None:
        """Decode JPEG frames from the Freenove StreamingOutput."""
        last_bytes: bytes | None = None
        while self._running:
            try:
                with self._stream_output.condition:
                    # Wait up to 0.5 s for a new frame
                    self._stream_output.condition.wait(timeout=0.5)
                    jpg = self._stream_output.frame
            except Exception as exc:
                logger.warning("StreamingOutput wait error: %s", exc)
                time.sleep(0.05)
                continue

            if jpg is None or jpg is last_bytes:
                continue
            last_bytes = jpg

            frame = _decode_jpeg(jpg, self._ai_size)
            if frame is not None:
                with self._lock:
                    self._buf.append(frame)
                    self._seq += 1

    def _demo_loop(self) -> None:
        """Read frames from a video file and feed the buffer at video frame-rate."""
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = 1.0 / fps
        while self._running:
            ok, bgr = self._cap.read()
            if not ok:
                # Loop video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bgr = self._cap.read()
                if not ok:
                    break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self._ai_size, self._ai_size))
            with self._lock:
                self._buf.append(resized)
                self._seq += 1

            time.sleep(delay)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_jpeg(jpg: bytes, target_size: int) -> np.ndarray | None:
    """Decode JPEG bytes to a resized RGB uint8 numpy array."""
    try:
        arr = np.frombuffer(jpg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (target_size, target_size))
    except Exception as exc:
        logger.debug("JPEG decode error: %s", exc)
        return None
