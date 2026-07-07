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
    # Bare-placeholder forms.
    ("Moving something closer", "person", "Moving person closer"),
    ("Pushing something from left to right", "chair", "Pushing chair from left to right"),
    ("Something falling like a rock", "bottle", "Bottle falling like a rock"),
    ("", "person", ""),                                            # empty template
    # Real SSv2/VideoMAE bracketed placeholders.
    ("Showing [something] behind [something]", "person", "Showing person behind person"),
    ("Moving [something] closer to [something]", "cup", "Moving cup closer to cup"),
    ("[Something] falling like a rock", "bottle", "Bottle falling like a rock"),
    # No object detected → strip the brackets, keep "something".
    ("Showing [something] behind [something]", "", "Showing something behind something"),
    ("Moving something closer", "", "Moving something closer"),
])
def test_fill_template(template, obj, expected):
    from ssv2_model import fill_template
    assert fill_template(template, obj) == expected


@pytest.mark.parametrize("template,obj,fallback,expected", [
    # No YOLO object → the fallback noun fills the slot (no leaked "something").
    ("Showing [something] behind [something]", "", "obstacle", "Showing obstacle behind obstacle"),
    ("Moving something closer", "", "obstacle", "Moving obstacle closer"),
    ("[Something] falling like a rock", "", "wall", "Wall falling like a rock"),
    # A real YOLO label still wins over the fallback.
    ("Moving something closer", "person", "obstacle", "Moving person closer"),
    # Empty fallback → old behaviour (keep "something").
    ("Moving something closer", "", "", "Moving something closer"),
])
def test_fill_template_fallback(template, obj, fallback, expected):
    from ssv2_model import fill_template
    assert fill_template(template, obj, fallback) == expected


def test_recognizer_reads_unknown_object_label(cfg):
    from ssv2_model import SSv2Recognizer
    cfg["ssv2"]["unknown_object_label"] = "wall"
    assert SSv2Recognizer(cfg)._unknown_label == "wall"
    # default is "obstacle"
    cfg["ssv2"].pop("unknown_object_label", None)
    assert SSv2Recognizer(cfg)._unknown_label == "obstacle"


def test_stub_composes_sentence_without_model(cfg):
    """With transformers/model unavailable the stub still fills the object."""
    from ssv2_model import SSv2Recognizer
    cfg["ssv2"]["model_id"] = "does-not-exist/force-stub"
    r = SSv2Recognizer(cfg)
    r.load()
    clip = [np.zeros((224, 224, 3), np.uint8) for _ in range(cfg["ssv2"]["num_frames"])]
    res = r.recognize(clip, "person")
    assert res.buffer_ready and res.is_stub
    assert "person" in res.sentence.lower()


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


# ── Server YOLO picks the CLOSEST/largest obstacle's label (SSv2 filler) ──────

def test_detector_closest_label_is_largest_box():
    import numpy as np
    from detector import Detector

    class Box:
        def __init__(self, xyxy, cls, conf):
            self.xyxy = [np.array(xyxy)]
            self.cls = [cls]
            self.conf = [conf]

    class Res:
        names = {0: "person", 56: "chair"}
        # chair is small, person is large → closest_label must be "person"
        boxes = [Box([10, 10, 50, 50], 56, 0.9), Box([0, 0, 200, 200], 0, 0.9)]
        def __call__(self, *a, **k):
            return [self]

    d = Detector({"detector": {
        "model": "x", "confidence_threshold": 0.4, "iou_threshold": 0.45,
        "obstacle_classes": ["person", "chair"], "center_zone_ratio": 0.4,
        "close_area_threshold": 0.08, "run_every_n_frames": 1,
    }})
    d._model = Res()
    r = d.detect(np.zeros((300, 400, 3), np.uint8))
    assert r.closest_label == "person"
    assert set(r.labels) == {"person", "chair"}


# ── Server-side run-logging toggle ────────────────────────────────────────────

def test_pipeline_logging_toggle(cfg):
    from ai_pipeline import AIPipeline
    cfg["logging"]["enabled"] = False
    ai = AIPipeline(cfg=cfg)
    assert ai.is_logging_enabled() is False
    ai.set_logging_enabled(True)
    assert ai.is_logging_enabled() is True
    assert ai.get_state().logging_enabled is True
    ai.set_logging_enabled(False)
    assert ai.is_logging_enabled() is False


def test_pipeline_logging_initial_from_config(cfg):
    from ai_pipeline import AIPipeline
    cfg["logging"]["enabled"] = True
    assert AIPipeline(cfg=cfg).is_logging_enabled() is True




def test_resolve_logging_enabled_precedence(monkeypatch):
    import main_server

    class Args:
        def __init__(self, logging):
            self.logging = logging

    # --logging flag wins over env and config
    monkeypatch.setenv("NAV_LOGGING", "1")
    assert main_server._resolve_logging_enabled(Args("off"), {"logging": {"enabled": True}}) is False
    # env wins over config when no flag
    assert main_server._resolve_logging_enabled(Args(None), {"logging": {"enabled": False}}) is True
    monkeypatch.setenv("NAV_LOGGING", "0")
    assert main_server._resolve_logging_enabled(Args(None), {"logging": {"enabled": True}}) is False
    # config used when neither flag nor env
    monkeypatch.delenv("NAV_LOGGING", raising=False)
    assert main_server._resolve_logging_enabled(Args(None), {"logging": {"enabled": True}}) is True
    assert main_server._resolve_logging_enabled(Args(None), {"logging": {"enabled": False}}) is False
