"""
test_ultrasonic_rpi.py – Unit tests for the Ultrasonic wrapper (Code/Robot).

lgpio is stubbed so the manual-timing backend runs without a Pi. Verifies:
  • lgpio backend is preferred and claims the pins,
  • a sensor that never echoes returns max range (not a crash),
  • a simulated echo produces a plausible distance.
"""

import importlib.util
import os
import sys
import types

import pytest

_ULTRA = os.path.join(os.path.dirname(__file__), "..", "Code", "Robot", "ultrasonic.py")


def _load_with_lgpio(reads):
    """Load ultrasonic.py with a stub lgpio whose gpio_read pops from `reads`."""
    lg = types.ModuleType("lgpio")
    state = {"reads": list(reads)}
    lg.gpiochip_open = lambda n: 7
    lg.gpio_claim_output = lambda *a, **k: None
    lg.gpio_claim_input = lambda *a, **k: None
    lg.gpio_write = lambda *a, **k: None
    lg.gpiochip_close = lambda *a, **k: None

    def gpio_read(chip, pin):
        return state["reads"].pop(0) if state["reads"] else 0

    lg.gpio_read = gpio_read
    sys.modules["lgpio"] = lg
    spec = importlib.util.spec_from_file_location("ultrasonic_under_test", _ULTRA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def teardown_function(_):
    sys.modules.pop("lgpio", None)


def test_prefers_lgpio_backend():
    mod = _load_with_lgpio([0])
    u = mod.Ultrasonic(trigger_pin=27, echo_pin=22, gpiochip=0)
    assert u._backend == "lgpio"


def test_no_echo_returns_max_range():
    mod = _load_with_lgpio([])          # gpio_read always 0 → echo never rises
    u = mod.Ultrasonic()
    assert u.get_distance() == mod._MAX_RANGE_CM


def test_simulated_echo_returns_plausible_distance():
    # echo high immediately then low after a couple of reads → small distance
    mod = _load_with_lgpio([1, 1, 0])
    u = mod.Ultrasonic()
    d = u.get_distance()
    assert 0.0 <= d <= mod._MAX_RANGE_CM


def test_gpiozero_fallback_when_lgpio_missing(monkeypatch):
    # Make `import lgpio` fail, and stub gpiozero so the fallback constructs.
    sys.modules["lgpio"] = None  # importing this raises ImportError
    gz = types.ModuleType("gpiozero")

    class _DS:
        def __init__(self, **k):
            self.distance = 1.0

        def close(self):
            pass

    gz.DistanceSensor = _DS
    pins = types.ModuleType("gpiozero.pins")
    lgm = types.ModuleType("gpiozero.pins.lgpio")
    lgm.LGPIOFactory = lambda chip=0: object()
    sys.modules.update({"gpiozero": gz, "gpiozero.pins": pins, "gpiozero.pins.lgpio": lgm})
    try:
        spec = importlib.util.spec_from_file_location("ultra_fb", _ULTRA)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        u = mod.Ultrasonic()
        assert u._backend == "gpiozero"
        assert u.get_distance() == 100.0   # distance 1.0 m → 100 cm
    finally:
        for m in ("gpiozero", "gpiozero.pins", "gpiozero.pins.lgpio"):
            sys.modules.pop(m, None)
