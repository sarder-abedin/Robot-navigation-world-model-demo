"""Tests for CameraBuffer in demo mode (no picamera2 required)."""

import sys, os, time, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

import numpy as np
import pytest
import yaml


@pytest.fixture
def cfg_demo(tmp_path):
    """Config pointing at a tiny synthetic video file."""
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "demo"
    # Create a tiny 10-frame 64×64 synthetic video
    video_path = str(tmp_path / "test_corridor.mp4")
    import cv2
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10, (64, 64),
    )
    for i in range(20):
        frame = np.full((64, 64, 3), i * 10, dtype=np.uint8)
        out.write(frame)
    out.release()
    cfg["camera"]["demo_video_path"] = video_path
    cfg["camera"]["ai_frame_size"] = 64
    cfg["camera"]["clip_length"] = 4
    return cfg


def test_buffer_starts_empty(cfg_demo):
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_demo)
    buf.start()
    # Wait briefly for the demo loop to fill the buffer
    time.sleep(0.5)
    assert len(buf) > 0
    buf.stop()


def test_get_latest_frame(cfg_demo):
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_demo)
    buf.start()
    time.sleep(0.5)
    frame = buf.get_latest_frame()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    buf.stop()


def test_get_clip_returns_none_before_full(cfg_demo):
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_demo)
    # Don't start – buffer is empty
    result = buf.get_clip(n=cfg_demo["camera"]["clip_length"])
    assert result is None


def test_get_clip_after_fill(cfg_demo):
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_demo)
    buf.start()
    time.sleep(0.8)  # Let buffer fill
    clip = buf.get_clip(n=cfg_demo["camera"]["clip_length"])
    assert clip is not None
    assert len(clip) == cfg_demo["camera"]["clip_length"]
    buf.stop()


@pytest.fixture
def cfg_tcp(tmp_path):
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "tcp"
    cfg["camera"]["ai_frame_size"] = 64
    cfg["camera"]["clip_length"] = 4
    return cfg


def test_tcp_mode_starts_without_thread(cfg_tcp):
    """In tcp mode start() should not launch a capture thread."""
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_tcp)
    buf.start()
    # No thread started; buffer starts empty
    assert len(buf) == 0
    buf.stop()


def test_push_frame_fills_buffer(cfg_tcp):
    """push_frame() with valid JPEG bytes appends frames to the buffer."""
    import cv2
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_tcp)
    buf.start()

    # Encode a small synthetic JPEG
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, jpg_buf = cv2.imencode(".jpg", img)
    jpg_bytes = jpg_buf.tobytes()

    assert len(buf) == 0
    buf.push_frame(jpg_bytes)
    assert len(buf) == 1
    frame = buf.get_latest_frame()
    assert frame is not None
    assert frame.shape == (64, 64, 3)
    buf.stop()


def test_push_frame_invalid_bytes_ignored(cfg_tcp):
    """push_frame() with non-JPEG bytes should not raise and not append."""
    from camera_buffer import CameraBuffer
    buf = CameraBuffer(cfg_tcp)
    buf.start()

    buf.push_frame(b"not_a_jpeg")
    assert len(buf) == 0
    buf.stop()
