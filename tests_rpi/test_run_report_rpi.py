"""
test_run_report_rpi.py – offline run-report plotting/summary (matplotlib, no GUI).

Covers load_run, summary, figure building and PNG export against a synthetic run,
plus graceful handling of an older CSV missing the newer depth/latency/network cols.
"""

import csv
import os
import sys

import numpy as np

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

import run_report as rr

FULL_COLS = [
    "timestamp", "frame_idx", "nav_mode", "action", "risk_score", "detector_risk",
    "world_model_risk", "temporal_risk", "wm_label", "temporal_pattern", "obstacles",
    "in_center", "closest_area", "ultrasonic_cm", "ssv2", "explanation",
    "depth_center_m", "depth_left_m", "depth_right_m", "lat_total_ms", "lat_yolo_ms",
    "lat_wm_ms", "lat_depth_ms", "lat_temporal_ms", "lat_ssv2_ms", "lat_decision_ms",
    "reaction_ema_ms", "net_recv_fps", "net_frame_bytes", "net_frames_recv",
    "net_frames_dropped", "net_kbps",
]


def _write_run(tmp_path, n=40, cols=FULL_COLS):
    run = tmp_path / "run_test"
    run.mkdir()
    rows = []
    for i in range(n):
        act = "FORWARD" if i % 2 == 0 else "STOP"
        r = {c: 0 for c in cols}
        r.update(timestamp=round(1000 + i * 0.1, 2), frame_idx=i + 1, nav_mode="predictive",
                 action=act, risk_score=0.2, detector_risk=0.0, world_model_risk=0.49,
                 temporal_risk=0.0, wm_label="MIXED", ultrasonic_cm=200 - i,
                 depth_center_m=1.5, lat_total_ms=20 + (80 if i % 8 == 0 else 0),
                 net_recv_fps=30, net_frames_recv=(i + 1) * 2, net_frames_dropped=i)
        rows.append({k: r.get(k, "") for k in cols})
    with open(run / "navigation_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    return str(run)


def test_load_run_relative_time_and_columns(tmp_path):
    d = rr.load_run(_write_run(tmp_path, n=10))
    assert len(d["t"]) == 10
    assert d["t"][0] == 0.0                                   # relative time starts at 0
    assert np.isclose(d["t"][-1], 0.9, atol=0.01)
    assert d["ultrasonic_m"][0] == 2.0                        # cm → m


def test_summary_numbers(tmp_path):
    d = rr.load_run(_write_run(tmp_path, n=40))
    s = rr.summary(d)
    assert s["frames"] == 40
    assert s["action_pct"]["FORWARD"] == 50.0 and s["action_pct"]["STOP"] == 50.0
    assert s["wm_risk_mean"] == 0.49 and s["wm_risk_std"] < 0.03   # flat V-JEPA2
    assert "uncalibrated" in rr.summary_text(d)                    # flags the flatness


def test_build_all_figures(tmp_path):
    d = rr.load_run(_write_run(tmp_path))
    figs = rr.build_all_figures(d)
    assert set(figs) == {"risk", "distance", "actions", "latency", "network"}
    for fig in figs.values():
        assert fig.axes                                            # each has content


def test_save_pngs_writes_files(tmp_path):
    run = _write_run(tmp_path)
    paths = rr.save_pngs(run)
    names = {os.path.basename(p) for p in paths}
    assert {"risk.png", "distance.png", "actions.png", "latency.png",
            "network.png", "summary.txt"} <= names
    for p in paths:
        assert os.path.getsize(p) > 0


def test_older_csv_without_new_columns(tmp_path):
    old_cols = ["timestamp", "frame_idx", "nav_mode", "action", "risk_score",
                "detector_risk", "world_model_risk", "temporal_risk", "wm_label",
                "temporal_pattern", "obstacles", "in_center", "closest_area",
                "ultrasonic_cm", "ssv2", "explanation"]
    d = rr.load_run(_write_run(tmp_path, n=12, cols=old_cols))
    # missing depth/latency/network → NaN arrays, figures still build, save works
    figs = rr.build_all_figures(d)
    assert set(figs) == {"risk", "distance", "actions", "latency", "network"}
    paths = rr.save_pngs(str(tmp_path / "run_test"))
    assert any(p.endswith("risk.png") for p in paths)
