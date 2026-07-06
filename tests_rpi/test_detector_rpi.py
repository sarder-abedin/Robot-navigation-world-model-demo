"""
test_detector_rpi.py – Detector obstacle-class filtering.

Constructs the Detector (no model load needed) and checks how obstacle_classes
resolves: an explicit list keeps a filter set, while "all"/"*"/[] disable the
filter so every YOLO class is treated as an obstacle.
"""

import os
import sys

import pytest
import yaml

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from detector import Detector


def _cfg(classes):
    return {
        "detector": {
            "model": "yolo11n.pt",
            "confidence_threshold": 0.4,
            "iou_threshold": 0.45,
            "obstacle_classes": classes,
            "center_zone_ratio": 0.4,
            "close_area_threshold": 0.08,
            "run_every_n_frames": 2,
        }
    }


def test_config_lists_all_80_coco_classes():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert len(cfg["detector"]["obstacle_classes"]) == 80


def test_explicit_classes_build_a_filter_set():
    d = Detector(_cfg(["person", "chair"]))
    assert d._obstacle_classes == {"person", "chair"}


@pytest.mark.parametrize("classes", [["all"], ["*"], ["ALL"], []])
def test_wildcard_or_empty_disables_the_filter(classes):
    d = Detector(_cfg(classes))
    assert d._obstacle_classes is None   # None → match every detected class


def test_full_80_class_list_matches_everything_it_lists():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    d = Detector(cfg)
    # Explicit 80-class list → a concrete set covering the whole COCO taxonomy.
    assert d._obstacle_classes is not None
    for name in ("person", "car", "dog", "toothbrush", "refrigerator"):
        assert name in d._obstacle_classes
