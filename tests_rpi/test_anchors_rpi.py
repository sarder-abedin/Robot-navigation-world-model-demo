"""
test_anchors_rpi.py – V-JEPA 2 corridor-anchor persistence + calibration helper.

Tests the save/load roundtrip directly (no encoder → no torch import, which keeps
the shared torch runtime stable for the rest of the suite) and the image loader.
The build-anchors-from-frames path is exercised by the world_model tests.
"""

import os
import sys

import numpy as np
import pytest
import yaml

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)


@pytest.fixture
def cfg():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        return yaml.safe_load(f)


def test_save_load_anchors_roundtrip(cfg, tmp_path):
    from world_model import WorldModel

    wm = WorldModel(cfg)
    # Set anchors directly (skip the encoder) and persist them.
    wm._obstacle_anchor = np.arange(1024, dtype=np.float32)
    wm._clear_anchor = np.arange(1024, dtype=np.float32)[::-1].copy()
    out = str(tmp_path / "anchors.npz")
    wm.save_anchors(out)
    assert os.path.exists(out)

    # A fresh model loads exactly those anchors back.
    wm2 = WorldModel(cfg)
    wm2.load_anchors(out)
    assert np.allclose(wm2._obstacle_anchor, wm._obstacle_anchor)
    assert np.allclose(wm2._clear_anchor, wm._clear_anchor)


def test_save_before_build_raises(cfg, tmp_path):
    from world_model import WorldModel

    wm = WorldModel(cfg)  # anchors not built yet
    with pytest.raises(RuntimeError):
        wm.save_anchors(str(tmp_path / "x.npz"))


def test_anchors_path_config_default_is_empty(cfg):
    # Default config ships with no calibrated anchors (synthetic fallback).
    assert cfg["world_model"].get("anchors_path", "") == ""


def test_calibrate_load_images(tmp_path):
    import cv2
    from calibrate_anchors import load_images

    d = tmp_path / "imgs"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.zeros((10, 10, 3), np.uint8))
    cv2.imwrite(str(d / "b.jpg"), np.full((10, 10, 3), 255, np.uint8))
    (d / "notes.txt").write_text("ignore me")
    imgs = load_images(str(d))
    assert len(imgs) == 2 and all(im.ndim == 3 for im in imgs)
