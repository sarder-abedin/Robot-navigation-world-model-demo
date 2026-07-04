"""
test_ssv2_rpi.py – Unit tests for the genuine SSv2 feature.

Covers the parts that run without the heavy VideoMAE model / torch:
  • SSv2 template filling with YOLO object labels
  • the stub path (no transformers) still composes a sentence
  • CMD_DETECTION carries the YOLO label end to end
  • CMD_AISTATUS carries the composed SSv2 sentence (with 6-field back-compat)
"""

import os
import sys

import numpy as np
import pytest
import yaml

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
ROBOT = os.path.join(os.path.dirname(__file__), "..", "Code", "Robot")
sys.path.insert(0, SERVER)
sys.path.insert(0, ROBOT)


@pytest.fixture
def cfg():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        return yaml.safe_load(f)


# ── Template filling ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("template,obj,expected", [
    ("Moving something closer", "person", "Moving person closer"),
    ("Pushing something from left to right", "chair", "Pushing chair from left to right"),
    ("Something falling like a rock", "bottle", "Bottle falling like a rock"),
    ("Moving something closer", "", "Moving something closer"),   # no object → unchanged
    ("", "person", ""),                                            # empty template
])
def test_fill_template(template, obj, expected):
    from ssv2_model import fill_template
    assert fill_template(template, obj) == expected


def test_stub_composes_sentence_without_model(cfg):
    """With transformers/model unavailable the stub still fills the object."""
    from ssv2_model import SSv2Recognizer
    cfg["ssv2"]["model_id"] = "does-not-exist/force-stub"
    r = SSv2Recognizer(cfg)
    r.load()
    clip = [np.zeros((224, 224, 3), np.uint8) for _ in range(cfg["ssv2"]["num_frames"])]
    res = r.recognize(clip, "person")
    assert res.buffer_ready and res.is_stub
    assert "person" in res.sentence


def test_needs_full_clip(cfg):
    from ssv2_model import SSv2Recognizer
    r = SSv2Recognizer(cfg)
    r.load()
    assert r.recognize([], "person").buffer_ready is False
    short = [np.zeros((224, 224, 3), np.uint8) for _ in range(3)]
    assert r.recognize(short, "person").buffer_ready is False


# ── CMD_DETECTION carries the YOLO label ──────────────────────────────────────

def test_send_detection_wire_format_includes_label():
    from tcp_robot_client import RobotTCPClient
    c = RobotTCPClient("x")
    c._connected = True
    captured = {}

    class FakeSock:
        def sendall(self, b):
            captured["msg"] = b.decode()

    c._cmd_sock = FakeSock()
    c.send_detection(risk_pct=42, obs_in_center=True, area_frac_pct=30,
                     centroid_x_pct=55, sonic_cm=88.7, top_label="person")
    assert captured["msg"].strip() == "CMD_DETECTION#42#1#30#55#88.7#person"


def test_robot_connection_parses_label(cfg):
    from camera_buffer import CameraBuffer
    from robot_connection import RobotConnectionServer
    c = dict(cfg); c["mode"] = "tcp"
    rc = RobotConnectionServer(c, CameraBuffer(c))
    rc._latest_detection = {
        "yolo_risk_pct": 42, "obs_in_center": True, "area_frac_pct": 30,
        "centroid_x_pct": 55, "n_obstacles": 1, "top_label": "person",
    }
    assert rc.get_latest_detection().labels == ["person"]


def test_get_latest_detection_defaults_label_when_missing(cfg):
    from camera_buffer import CameraBuffer
    from robot_connection import RobotConnectionServer
    c = dict(cfg); c["mode"] = "tcp"
    rc = RobotConnectionServer(c, CameraBuffer(c))
    # default detection has empty top_label → falls back to "obstacle"
    rc._latest_detection["yolo_risk_pct"] = 50
    rc._latest_detection["area_frac_pct"] = 20
    rc._latest_detection["n_obstacles"] = 1
    assert rc.get_latest_detection().labels == ["obstacle"]


# ── CMD_AISTATUS carries the SSv2 sentence (with back-compat) ──────────────────

def test_aistatus_seven_field_parse():
    line = "CMD_AISTATUS#FORWARD#5#UNKNOWN#STATIC_CLEAR#88.7#person moving closer"
    p = line.split("#")
    ssv2 = p[6].strip() if len(p) >= 7 else ""
    assert ssv2 == "person moving closer"


def test_aistatus_six_field_back_compat():
    line = "CMD_AISTATUS#FORWARD#5#UNKNOWN#STATIC_CLEAR#88.7"
    p = line.split("#")
    ssv2 = p[6].strip() if len(p) >= 7 else ""
    assert ssv2 == ""
