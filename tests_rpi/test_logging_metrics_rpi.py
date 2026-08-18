"""
test_logging_metrics_rpi.py – the run log carries inference-latency + network stats.

Checks that NavigationLogger writes the metric columns (per-stage latency, reaction
EMA, and camera-stream network statistics) and that CameraBuffer.get_net_stats
reports frames/bytes/fps for the received JPEG stream.
"""

import csv
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from ai_logger import METRIC_FIELDS, NavigationLogger


@dataclass
class _Decision:
    action: str = "FORWARD"
    risk_score: float = 0.12
    detector_risk: float = 0.0
    world_model_risk: float = 0.48
    temporal_risk: float = 0.0
    world_model_label: str = "MIXED"
    temporal_pattern: str = "STATIC_CLEAR"
    explanation: str = "Low risk – forward"


@dataclass
class _Det:
    boxes: list = field(default_factory=list)
    obstacle_in_center: bool = False
    closest_area: float = 0.0


def _cfg(log_dir):
    return {"logging": {
        "log_dir": str(log_dir),
        "save_annotated_frames": False,
        "csv_log": True,
        "log_level": "INFO",
    }}


def test_log_frame_writes_latency_and_network_columns(tmp_path):
    nav = NavigationLogger(_cfg(tmp_path), "predictive")
    metrics = {
        "lat_total_ms": 123.4, "lat_yolo_ms": 20.1, "lat_wm_ms": 80.0,
        "lat_depth_ms": 15.0, "lat_temporal_ms": 0.3, "lat_ssv2_ms": 5.0,
        "lat_decision_ms": 0.2, "reaction_ema_ms": 110.0,
        "net_recv_fps": 12.5, "net_frame_bytes": 8257, "net_frames_recv": 42,
        "net_frames_dropped": 3, "net_kbps": 826.0,
    }
    nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det(),
                  ultrasonic_cm=62.2, ssv2_sentence="approaching", metrics=metrics)
    nav.close()

    csv_path = next(tmp_path.rglob("navigation_log.csv"))
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    for col in METRIC_FIELDS:
        assert col in row                      # header carries every metric column
    assert row["lat_total_ms"] == "123.40"
    assert row["lat_wm_ms"] == "80.00"
    assert row["net_frame_bytes"] == "8257"    # byte/frame counts are integers
    assert row["net_frames_dropped"] == "3"
    assert row["net_recv_fps"] == "12.50"


def test_log_frame_saves_raw_and_depth_columns(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["logging"]["save_raw_frames"] = True
    cfg["logging"]["raw_frame_interval"] = 1
    nav = NavigationLogger(cfg, "predictive")
    raw = np.full((4, 4, 3), 7, np.uint8)
    nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det(),
                  raw_frame=raw, depth={"center": 0.45, "left": 0.4, "right": 0.9})
    nav.close()
    run_dir = next(tmp_path.rglob("navigation_log.csv")).parent
    # raw frame written to the SEPARATE raw_frames/ folder
    assert list((run_dir / "raw_frames").glob("*.jpg"))
    with open(run_dir / "navigation_log.csv") as f:
        row = next(csv.DictReader(f))
    assert row["depth_center_m"] == "0.450" and row["depth_right_m"] == "0.900"


def test_log_frame_without_metrics_defaults_to_zero(tmp_path):
    nav = NavigationLogger(_cfg(tmp_path), "baseline")
    nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det())  # no metrics
    nav.close()
    csv_path = next(tmp_path.rglob("navigation_log.csv"))
    with open(csv_path) as f:
        row = next(csv.DictReader(f))
    assert row["lat_total_ms"] == "0.00" and row["net_frames_recv"] == "0"


def test_log_frame_after_close_is_safe_noop(tmp_path):
    # The drive loop can call log_frame() while stop() is closing the logger.
    # After close() a late write must NOT raise "I/O operation on closed file".
    nav = NavigationLogger(_cfg(tmp_path), "predictive")
    nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det())
    nav.close()
    # Late write after close → silently dropped, no exception.
    nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det())
    csv_path = next(tmp_path.rglob("navigation_log.csv"))
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1                      # only the pre-close row landed


def test_concurrent_log_and_close_never_raises(tmp_path):
    # Race a writer thread against close() the way stop() does; the writer must
    # never hit a closed file (the lock + writer=None guard prevents it).
    import threading
    nav = NavigationLogger(_cfg(tmp_path), "predictive")
    errors = []

    def writer():
        for _ in range(500):
            try:
                nav.log_frame(np.zeros((4, 4, 3), np.uint8), _Decision(), _Det())
            except Exception as exc:            # ValueError on a closed file, etc.
                errors.append(exc)
                return

    t = threading.Thread(target=writer)
    t.start()
    nav.close()
    t.join()
    assert not errors, f"log_frame raced with close(): {errors[:1]}"


def test_camera_buffer_net_stats():
    import cv2
    from camera_buffer import CameraBuffer

    cfg = {"mode": "tcp", "camera": {
        "clip_length": 4, "ai_frame_size": 64,
        "demo_video_path": "", "stream_width": 64, "stream_height": 64,
    }}
    buf = CameraBuffer(cfg)
    ok, jpg = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    assert ok
    for _ in range(3):
        buf.push_frame(jpg.tobytes())
    stats = buf.get_net_stats()
    assert stats["frames_recv"] == 3
    assert stats["last_bytes"] == len(jpg.tobytes())
    assert stats["total_bytes"] == 3 * len(jpg.tobytes())
    assert stats["recv_fps"] >= 0.0
