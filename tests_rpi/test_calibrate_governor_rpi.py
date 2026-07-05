"""
test_calibrate_governor_rpi.py – governor calibration helpers + safe config patch.

Covers the pure logic (speed/decel/sanity/block) and patch_config_governor's
"safe and reliable" contract: it edits only the governor numerics, preserves
comments, keeps the YAML valid, and raises (leaving the file usable) on bad input.
The hardware-driving main() isn't unit-tested (needs motor + ultrasonic).
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Robot"))

import calibrate_governor as cg


def test_speed_from_samples_slope():
    # 1 m/s: distance drops 0.1 m every 0.1 s
    samples = [(i * 0.1, 2.0 - i * 0.1) for i in range(6)]
    assert abs(cg.speed_from_samples(samples) - 1.0) < 1e-6


def test_speed_from_samples_needs_two_points():
    assert cg.speed_from_samples([(0.0, 1.0)]) == 0.0


def test_decel_from_coast():
    # v=1 m/s, coast 0.5 m → a = 1/(2*0.5) = 1.0
    assert abs(cg.decel_from_coast(1.0, 0.5) - 1.0) < 1e-9
    assert cg.decel_from_coast(1.0, 0.0) == 0.0


def test_sanity_flags_bad_values():
    assert cg.sanity_problems(0.0, 0.1, 0.5)          # no forward motion
    assert cg.sanity_problems(0.1, 0.3, 0.5)          # slow faster than forward
    assert cg.sanity_problems(0.35, 0.18, 0.0)        # no decel
    assert not cg.sanity_problems(0.35, 0.18, 0.6)    # plausible → no problems


_CFG = """\
decision:
  weights:
    detector: 0.35
  low_risk_max: 0.25
  governor:
    enabled: true
    forward_speed_mps: 0.35    # measured
    slow_speed_mps: 0.18       # measured
    max_decel_mps2: 0.6        # measured
    safety_margin_m: 0.10
robot:
  speed_full: 1500
"""


def test_patch_updates_only_governor_numerics_and_keeps_comments(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_CFG)
    backup = cg.patch_config_governor(str(p), {
        "forward_speed_mps": "0.412", "slow_speed_mps": "0.201", "max_decel_mps2": "0.55",
    })
    assert os.path.exists(backup)
    text = p.read_text()
    # comments preserved, other keys untouched, YAML still valid with new values
    assert "# measured" in text
    assert "safety_margin_m: 0.10" in text
    cfg = yaml.safe_load(text)
    g = cfg["decision"]["governor"]
    assert abs(g["forward_speed_mps"] - 0.412) < 1e-9
    assert abs(g["slow_speed_mps"] - 0.201) < 1e-9
    assert abs(g["max_decel_mps2"] - 0.55) < 1e-9
    assert g["enabled"] is True                        # untouched
    assert cfg["robot"]["speed_full"] == 1500          # untouched


def test_patch_raises_when_no_governor_block(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("decision:\n  low_risk_max: 0.25\n")
    with pytest.raises(ValueError):
        cg.patch_config_governor(str(p), {"forward_speed_mps": "0.4"})


def test_patch_raises_when_key_missing(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("decision:\n  governor:\n    enabled: true\n")
    with pytest.raises(ValueError):
        cg.patch_config_governor(str(p), {"forward_speed_mps": "0.4"})


def test_config_block_renders():
    block = cg.config_block({"forward_speed_mps": 0.35, "slow_speed_mps": 0.18, "max_decel_mps2": 0.6})
    assert "governor:" in block and "forward_speed_mps: 0.350" in block
