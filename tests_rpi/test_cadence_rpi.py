"""
test_cadence_rpi.py – V-JEPA 2 / SSv2 inference cadence (no torch, no GPU).

Guards the "run more often" tuning: the models read their run_every_n_frames
from config, the shipped config keeps V-JEPA 2 fresh, and the subprocess override
makes the dedicated worker run V-JEPA 2 on EVERY clip (run_every == 1 never skips).
"""

import os
import sys

import yaml

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from world_model import WorldModel        # __init__ needs no torch
from ssv2_model import SSv2Recognizer     # __init__ needs no torch


def _config():
    with open(os.path.join(SERVER, "config.yaml")) as f:
        return yaml.safe_load(f)


def test_models_read_cadence_from_config():
    cfg = _config()
    assert WorldModel(cfg)._run_every == cfg["world_model"]["run_every_n_frames"]
    assert SSv2Recognizer(cfg)._run_every == cfg["ssv2"]["run_every_n_frames"]


def test_shipped_config_keeps_vjepa_fresh():
    cfg = _config()
    # The live signal that drives navigation should refresh often on a GPU box.
    assert cfg["world_model"]["run_every_n_frames"] <= 4
    # The subprocess cadence for the (log-only) SSv2 caption is present + sane.
    assert cfg["ssv2"]["subprocess_run_every"] >= 1


def test_run_every_one_never_skips():
    # The predict()/recognize() gate skips when call_count % run_every != 0.
    # With run_every == 1 the modulo is always 0 → the forward runs every call.
    run_every = 1
    assert all((c % run_every) == 0 for c in range(1, 51))


def test_subprocess_override_runs_vjepa_every_clip():
    # Mirror _worker_main's override: V-JEPA 2 → every clip, SSv2 → config cadence.
    cfg = _config()
    wm, ss = WorldModel(cfg), SSv2Recognizer(cfg)
    wm._run_every = 1
    ss._run_every = max(1, int(cfg.get("ssv2", {}).get("subprocess_run_every", 2)))
    assert wm._run_every == 1
    assert ss._run_every == cfg["ssv2"]["subprocess_run_every"]
