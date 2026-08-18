"""
test_build_anchor_set_rpi.py – balanced blocked/clear anchor-set extraction.

Covers the pure labelling/balancing and the logged-run extraction path (which
reuses calibrate_from_logs' independent auto-labelling). No torch/YOLO — the
video path needs the real model and isn't exercised here.
"""

import csv
import os
import sys

import cv2
import numpy as np

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

import build_anchor_set as bas


def test_label_by_detection_thresholds():
    assert bas.label_by_detection(True, 0.20) == "blocked"      # centred + large
    assert bas.label_by_detection(True, 0.05) is None           # centred but small → ambiguous
    assert bas.label_by_detection(False, 0.30) is None          # large but off-centre
    assert bas.label_by_detection(False, 0.01) == "clear"       # nothing significant
    # custom thresholds
    assert bas.label_by_detection(True, 0.08, block_area=0.05) == "blocked"


def test_balance_trims_to_min_and_caps():
    b, c = bas.balance(list(range(10)), list(range(4)), per_class=24)
    assert len(b) == len(c) == 4                                # min of the two
    b, c = bas.balance(list(range(10)), list(range(10)), per_class=3)
    assert len(b) == len(c) == 3                                # capped
    assert b == [0, 3, 6] and c == [0, 3, 6]                    # even spread
    assert bas.balance([], [1, 2], 5) == ([], [])              # one side empty → nothing


def _write_run(tmp_path, raw=True):
    run = tmp_path / "run_x"
    (run / ("raw_frames" if raw else "frames")).mkdir(parents=True)
    sub = "raw_frames" if raw else "frames"
    cols = ["frame_idx", "action", "detector_risk", "world_model_risk", "risk_score",
            "obstacles", "in_center", "ultrasonic_cm"]
    rows = []
    # 6 clearly-blocked frames (STOP + high det + centred + near sonar)
    for i in range(1, 7):
        rows.append({"frame_idx": i, "action": "STOP", "detector_risk": 0.7,
                     "world_model_risk": 0.5, "risk_score": 0.7, "obstacles": 2,
                     "in_center": 1, "ultrasonic_cm": 30})
    # 6 clearly-clear frames (FORWARD + low risk + no obstacle + far sonar)
    for i in range(7, 13):
        rows.append({"frame_idx": i, "action": "FORWARD", "detector_risk": 0.0,
                     "world_model_risk": 0.2, "risk_score": 0.1, "obstacles": 0,
                     "in_center": 0, "ultrasonic_cm": 150})
    with open(run / "navigation_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for i in range(1, 13):
        img = np.full((16, 16, 3), i * 5, np.uint8)
        cv2.imwrite(str(run / sub / f"frame_{i:06d}.jpg"), img)
    return str(run)


def test_frames_from_run_balanced_from_raw(tmp_path):
    run = _write_run(tmp_path, raw=True)
    blocked, clear, used_raw = bas.frames_from_run(run, per_class=4)
    assert used_raw is True
    assert len(blocked) == len(clear) == 4                      # balanced + capped
    # returns (name, bgr) pairs with real images
    assert all(name.startswith("frame_") and img is not None for name, img in blocked)
    # blocked names come from idx 1..6, clear from 7..12
    b_idx = [int(n.split("_")[1].split(".")[0]) for n, _ in blocked]
    c_idx = [int(n.split("_")[1].split(".")[0]) for n, _ in clear]
    assert all(1 <= i <= 6 for i in b_idx) and all(7 <= i <= 12 for i in c_idx)


def test_frames_from_run_falls_back_to_annotated(tmp_path):
    run = _write_run(tmp_path, raw=False)                       # only frames/, no raw_frames/
    blocked, clear, used_raw = bas.frames_from_run(run, per_class=10)
    assert used_raw is False                                    # caller warns on this
    assert len(blocked) == len(clear) == 6


def test_write_set_creates_folders(tmp_path):
    blocked = [("frame_000001.jpg", np.zeros((8, 8, 3), np.uint8))]
    clear = [("frame_000007.jpg", np.full((8, 8, 3), 200, np.uint8))]
    bdir, cdir = bas.write_set(str(tmp_path / "out"), blocked, clear)
    assert os.path.isfile(os.path.join(bdir, "frame_000001.jpg"))
    assert os.path.isfile(os.path.join(cdir, "frame_000007.jpg"))
