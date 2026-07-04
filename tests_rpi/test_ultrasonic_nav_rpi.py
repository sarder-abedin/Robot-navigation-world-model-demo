"""
test_ultrasonic_nav_rpi.py – server-side ultrasonic→hard-stop mapping.

Verifies _distance_to_risk (robot_control): a valid close reading trips the hard
stop (1.0), a far reading is clear (0.0), a momentary no-echo reuses the last
valid reading (so one dropped ping doesn't forget a close wall), and a sustained
no-echo releases to 0.0 (vision drives) rather than freezing or reading "clear".
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

from robot_control import _distance_to_risk


def test_close_reading_hard_stops():
    state = {"cm": None, "t": 0.0}
    assert _distance_to_risk(10.0, 30.0, state) == 1.0


def test_far_reading_is_clear():
    state = {"cm": None, "t": 0.0}
    assert _distance_to_risk(200.0, 30.0, state) == 0.0


def test_warn_zone_is_graded():
    state = {"cm": None, "t": 0.0}
    r = _distance_to_risk(60.0, 30.0, state)   # between stop (30) and 3×stop (90)
    assert 0.0 < r < 1.0


def test_blind_reuses_recent_close_reading():
    state = {"cm": None, "t": 0.0}
    _distance_to_risk(10.0, 30.0, state)              # remember a close wall
    # next ping misses; within the hold window we still report the hard stop
    assert _distance_to_risk(-1.0, 30.0, state, blind_hold_s=5.0) == 1.0


def test_blind_beyond_hold_releases():
    # last valid reading was long ago → no hard stop; let vision drive
    state = {"cm": 10.0, "t": time.monotonic() - 10.0}
    assert _distance_to_risk(-1.0, 30.0, state, blind_hold_s=1.0) == 0.0


def test_blind_with_no_history_is_not_a_stop():
    state = {"cm": None, "t": 0.0}
    assert _distance_to_risk(-1.0, 30.0, state, blind_hold_s=1.0) == 0.0
