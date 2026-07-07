"""
test_calibrate_from_logs_rpi.py – offline (zero-driving) calibration from logs.

Covers the pure functions of calibrate_from_logs.py: depth-scale from
sonar/depth pairs, governor speed from FORWARD/SLOW segments, blocked/clear
auto-labelling, and the surgical config patcher.
"""

import os
import sys

import pytest

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from calibrate_from_logs import (
    autolabel_rows, depth_scale_from_rows, governor_from_rows, patch_config_block,
)


def _row(**kw):
    base = dict(timestamp="0", frame_idx="1", action="FORWARD", risk_score="0.2",
                detector_risk="0.0", world_model_risk="0.48", obstacles="0",
                in_center="0", ultrasonic_cm="-1", depth_center_m="-1")
    base.update({k: str(v) for k, v in kw.items()})
    return base


# ── Depth scale ────────────────────────────────────────────────────────────────

def test_depth_scale_median_of_sonar_over_depth():
    # sonar reads 1.0 m, depth reports 1.25 m → scale should be ~0.8.
    rows = [_row(ultrasonic_cm=100, depth_center_m=1.25) for _ in range(30)]
    scale, n = depth_scale_from_rows(rows)
    assert n == 30 and scale == pytest.approx(0.8, abs=0.01)


def test_depth_scale_needs_enough_pairs_and_skips_blind_sonar():
    rows = [_row(ultrasonic_cm=-1, depth_center_m=1.0) for _ in range(30)]  # all blind
    scale, n = depth_scale_from_rows(rows)
    assert scale is None and n == 0


# ── Governor speeds ─────────────────────────────────────────────────────────────

def test_governor_forward_speed_from_decreasing_sonar():
    # 0.3 m/s: at 0.1 s intervals the wall distance drops 3 cm/step.
    rows = []
    for i in range(10):
        rows.append(_row(action="FORWARD", timestamp=round(i * 0.1, 2),
                         ultrasonic_cm=200 - i * 3))
    g = governor_from_rows(rows)
    assert g["forward_speed_mps"] == pytest.approx(0.30, abs=0.02)
    assert g["n_forward"] >= 1 and g["slow_speed_mps"] is None


def test_governor_ignores_stationary_or_blind_segments():
    rows = [_row(action="FORWARD", timestamp=round(i * 0.1, 2), ultrasonic_cm=200)
            for i in range(10)]                       # not moving (constant distance)
    g = governor_from_rows(rows)
    assert g["forward_speed_mps"] is None


# ── Auto-labelling ──────────────────────────────────────────────────────────────

def test_autolabel_blocked_and_clear():
    rows = [
        _row(frame_idx=1, action="FORWARD", risk_score=0.15, obstacles=0, ultrasonic_cm=200),  # clear
        _row(frame_idx=2, action="STOP", detector_risk=0.9, in_center=1, ultrasonic_cm=25),    # blocked
        _row(frame_idx=3, action="REROUTE", detector_risk=0.5, ultrasonic_cm=30),              # blocked
        _row(frame_idx=4, action="SLOW", risk_score=0.4, ultrasonic_cm=60),                    # ambiguous → skip
    ]
    blocked, clear = autolabel_rows(rows)
    assert set(blocked) == {2, 3} and set(clear) == {1}


# ── Config patch ────────────────────────────────────────────────────────────────

def test_patch_config_block_preserves_and_validates(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "depth:\n"
        "  scale: 1.0          # keep comment\n"
        "  max_range_m: 5.0\n"
        "world_model:\n"
        "  anchors_path: \"\"\n"
    )
    patch_config_block(str(cfg), "depth", {"scale": 0.74})
    patch_config_block(str(cfg), "world_model", {"anchors_path": "anchors.npz"})
    import yaml
    out = yaml.safe_load(cfg.read_text())
    assert out["depth"]["scale"] == 0.74
    assert out["depth"]["max_range_m"] == 5.0            # untouched
    assert out["world_model"]["anchors_path"] == "anchors.npz"
    assert "keep comment" in cfg.read_text()             # comment preserved


def test_patch_config_block_missing_block_raises(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("depth:\n  scale: 1.0\n")
    with pytest.raises(ValueError):
        patch_config_block(str(cfg), "governor", {"forward_speed_mps": 0.3})
