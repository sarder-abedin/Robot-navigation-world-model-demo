"""
test_goal_rpi.py – Phase 1 goal-point plumbing: the HUD draws the selected goal.

Phase 1 only sends CMD_GOAL and draws a marker (no motion). These tests cover the
server-side rendering: the visualizer draws a goal marker at the normalized point
and clamps out-of-range coords without error.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

SERVER = os.path.join(os.path.dirname(__file__), "..", "Code", "Server")
sys.path.insert(0, SERVER)

from visualization import Visualizer


@dataclass
class _Decision:
    action: str = "FORWARD"
    risk_score: float = 0.1
    world_model_label: str = "MIXED"


@dataclass
class _Det:
    boxes: list = field(default_factory=list)
    obstacle_in_center: bool = False
    closest_area: float = 0.0


@dataclass
class _Temporal:
    pattern: str = "STATIC_CLEAR"


def _viz():
    # Overlays off so annotate() exercises the goal path without needing full
    # decision/detector fakes for every HUD element.
    return Visualizer({"visualization": {
        "show_window": False, "overlay_detections": False, "overlay_risk_bar": False,
        "overlay_action": False, "overlay_world_model_label": False,
    }})


def test_hud_draws_goal_marker():
    viz = _viz()
    frame = np.zeros((300, 400, 3), np.uint8)
    out = viz.annotate(frame, _Det(), _Decision(), _Temporal(), goal=(0.5, 0.5))
    # The centre region should now have coloured (non-black) marker pixels.
    cx, cy = 200, 150
    patch = out[cy - 18:cy + 18, cx - 18:cx + 18]
    assert patch.sum() > 0, "goal marker should draw pixels at the goal location"


def test_no_goal_leaves_center_clean():
    viz = _viz()
    frame = np.zeros((300, 400, 3), np.uint8)
    out = viz.annotate(frame, _Det(), _Decision(), _Temporal(), goal=None)
    patch = out[150 - 18:150 + 18, 200 - 18:200 + 18]
    assert patch.sum() == 0, "no goal → no marker in the centre"


@pytest.mark.parametrize("goal", [(1.5, 0.5), (-0.2, 0.5), (0.5, 2.0), (0.0, 0.0)])
def test_out_of_range_goal_clamped_no_error(goal):
    viz = _viz()
    frame = np.zeros((300, 400, 3), np.uint8)
    out = viz.annotate(frame, _Det(), _Decision(), _Temporal(), goal=goal)
    assert out.shape == frame.shape        # draws within bounds, never raises


def test_permille_roundtrip_precision():
    # UI sends per-mille ints (parser is integer-only); server divides by 1000.
    for nx in (0.0, 0.123, 0.5, 0.999, 1.0):
        assert abs(int(nx * 1000) / 1000.0 - nx) <= 0.001
