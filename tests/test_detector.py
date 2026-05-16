"""
Tests for the detector module – exercises risk heuristic without loading YOLO.
"""

import numpy as np
import pytest
import yaml
from src.detector import Detector, DetectionResult


@pytest.fixture
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_no_detections_gives_zero_risk(cfg):
    detector = Detector(cfg)
    risk = detector._compute_raw_risk(False, 0.0, 0)
    assert risk == 0.0


def test_large_centered_obstacle_gives_high_risk(cfg):
    detector = Detector(cfg)
    # Area far above threshold, obstacle in center, multiple boxes
    risk = detector._compute_raw_risk(True, 0.20, 3)
    assert risk > 0.7


def test_off_center_small_obstacle_gives_low_risk(cfg):
    detector = Detector(cfg)
    risk = detector._compute_raw_risk(False, 0.01, 1)
    assert risk < 0.3


def test_risk_clamped_to_one(cfg):
    detector = Detector(cfg)
    risk = detector._compute_raw_risk(True, 1.0, 100)
    assert risk <= 1.0
