"""
calibrate_governor.py – measure the speed governor's SI constants on the robot.

The governor works in metres / seconds / m·s⁻¹ / m·s⁻², but the robot speaks PWM.
This tool measures the real values using the hardware you already have — the
motors and the forward ultrasonic — by pointing the robot at a flat wall and
using the sonar as a tape measure:

  • forward_speed_mps / slow_speed_mps – drive at the FORWARD / SLOW PWM and fit
    distance-vs-time from the ultrasonic → speed = -slope.
  • max_decel_mps2 – at the measured speed, command STOP and measure the coast
    distance d; a = v² / (2·d).

It prints a ready-to-paste `decision.governor` block and can optionally patch a
config file in place (safely: sanity-check → backup → surgical patch that keeps
comments → re-validate the YAML → restore the backup on any failure).

Run ON THE ROBOT (needs motor + ultrasonic), facing a flat wall with ~2–3 m of
clear runway:

  python calibrate_governor.py                       # measure + print the block
  python calibrate_governor.py --apply ../Server/config.yaml   # measure + patch

SAFETY: it aborts a run if the wall gets within --min-distance-cm, and refuses to
emit / apply values that aren't physically plausible.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time

import numpy as np
import yaml

_TARGET_KEYS = ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2")


# ── Pure helpers (unit-tested; no hardware) ──────────────────────────────────

def speed_from_samples(samples: list[tuple[float, float]]) -> float:
    """Speed (m/s) from (time_s, distance_m) samples while approaching a wall.

    Distance decreases as the robot advances, so speed = -slope of a linear fit.
    """
    if len(samples) < 2:
        return 0.0
    t = np.array([s[0] for s in samples], dtype=float)
    d = np.array([s[1] for s in samples], dtype=float)
    slope = float(np.polyfit(t, d, 1)[0])
    return max(0.0, -slope)


def decel_from_coast(v_mps: float, coast_m: float) -> float:
    """Deceleration (m/s²) to stop from v over a coast distance: a = v²/(2d)."""
    if coast_m <= 0 or v_mps <= 0:
        return 0.0
    return (v_mps * v_mps) / (2.0 * coast_m)


def sanity_problems(forward: float, slow: float, decel: float) -> list[str]:
    """Return a list of reasons the measured values are not physically plausible."""
    p = []
    if not forward > 0:
        p.append("forward_speed_mps must be > 0 (no motion measured — check the wall/sonar)")
    if not slow > 0:
        p.append("slow_speed_mps must be > 0")
    if forward > 0 and slow > 0 and not forward > slow:
        p.append("forward_speed_mps must exceed slow_speed_mps")
    if not decel > 0:
        p.append("max_decel_mps2 must be > 0 (no coast measured)")
    if forward > 3.0:
        p.append(f"forward_speed_mps={forward:.2f} implausibly fast (>3 m/s)")
    if decel > 30.0:
        p.append(f"max_decel_mps2={decel:.2f} implausibly high")
    return p


def config_block(values: dict) -> str:
    """Render the ready-to-paste governor YAML block."""
    return (
        "decision:\n"
        "  governor:\n"
        f"    forward_speed_mps: {values['forward_speed_mps']:.3f}\n"
        f"    slow_speed_mps: {values['slow_speed_mps']:.3f}\n"
        f"    max_decel_mps2: {values['max_decel_mps2']:.3f}\n"
    )


def patch_config_governor(path: str, updates: dict) -> str:
    """Surgically update the governor numerics in a YAML file, preserving comments.

    Backs up to <path>.bak, patches only the target keys inside the `governor:`
    block, re-parses to confirm the file is still valid YAML with the new values,
    and restores the backup on any failure. Returns the backup path.
    """
    with open(path) as f:
        lines = f.readlines()

    gov_idx = gov_indent = None
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)governor:\s*(#.*)?$", ln)
        if m:
            gov_idx, gov_indent = i, len(m.group(1))
            break
    if gov_idx is None:
        raise ValueError("no `governor:` block found in " + path)

    remaining = dict(updates)
    i = gov_idx + 1
    while i < len(lines) and remaining:
        ln = lines[i]
        if ln.strip() and not ln.lstrip().startswith("#"):
            indent = len(ln) - len(ln.lstrip())
            if indent <= gov_indent:
                break  # dedented out of the governor block
            m = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*([^#\n]*)(#.*)?$", ln)
            if m and m.group(2) in remaining:
                key = m.group(2)
                val = remaining.pop(key)
                comment = m.group(4) or ""
                new = f"{m.group(1)}{key}: {val}" + (f"  {comment}" if comment else "")
                lines[i] = new.rstrip() + "\n"
        i += 1
    if remaining:
        raise ValueError(f"keys not found in governor block: {sorted(remaining)}")

    backup = path + ".bak"
    shutil.copyfile(path, backup)
    with open(path, "w") as f:
        f.writelines(lines)
    try:
        cfg = yaml.safe_load(open(path))
        g = cfg["decision"]["governor"]
        for k, v in updates.items():
            if abs(float(g[k]) - float(v)) > 1e-6:
                raise ValueError(f"post-write mismatch for {k}")
    except Exception:
        shutil.copyfile(backup, path)  # restore — never leave a broken config
        raise
    return backup


# ── Hardware measurement (run on the robot) ──────────────────────────────────

def _sample_distance(ultrasonic, seconds: float, min_cm: float, motor=None):
    """Collect (t, distance_m) samples for `seconds`, aborting if too close."""
    samples, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        cm = ultrasonic.get_distance()
        if cm is not None and cm > 0:
            if cm < min_cm:
                if motor:
                    motor.setMotorModel(0, 0)
                raise RuntimeError(f"aborted: wall within {cm:.0f}cm (< {min_cm:.0f})")
            samples.append((time.perf_counter() - t0, cm / 100.0))
        time.sleep(0.02)
    return samples


def _measure_once(motor, ultrasonic, pwm, drive_s, min_cm):
    """Drive at `pwm`, return (speed_mps, coast_m) using the STOP coast."""
    motor.setMotorModel(pwm, pwm)
    time.sleep(0.3)                              # skip the soft-start ramp
    samples = _sample_distance(ultrasonic, drive_s, min_cm, motor)
    d_at_stop = samples[-1][1] if samples else -1.0
    motor.setMotorModel(0, 0)                    # STOP → coast
    time.sleep(1.2)                              # let it settle
    d_final = -1.0
    for _ in range(10):
        cm = ultrasonic.get_distance()
        if cm and cm > 0:
            d_final = cm / 100.0
        time.sleep(0.05)
    speed = speed_from_samples(samples)
    coast = max(0.0, d_at_stop - d_final) if d_at_stop > 0 and d_final > 0 else 0.0
    return speed, coast


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the speed governor on the robot")
    ap.add_argument("--config", default="config_robot.yaml", help="robot config (for PWM speeds)")
    ap.add_argument("--apply", default="", help="server config.yaml to patch in place (optional)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--drive-seconds", type=float, default=1.5)
    ap.add_argument("--min-distance-cm", type=float, default=30.0)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        rcfg = yaml.safe_load(f)
    robot = rcfg.get("robot", {})
    pwm_full = int(os.environ.get("SPEED_FULL", robot.get("speed_full", 1600)))
    pwm_slow = int(os.environ.get("SPEED_SLOW", robot.get("speed_slow", 1000)))
    gpio_chip = int(os.environ.get("GPIO_CHIP", rcfg.get("gpio", {}).get("chip", 0)))
    scfg = rcfg.get("ultrasonic", {})

    from motor import tankMotor
    from ultrasonic import Ultrasonic
    motor = tankMotor(gpiochip=gpio_chip, soft_start=robot.get("soft_start", True))
    ultra = Ultrasonic(trigger_pin=scfg.get("trigger_pin", 27),
                       echo_pin=scfg.get("echo_pin", 22), gpiochip=gpio_chip)

    print(f"Calibrating: face a flat wall with clear runway. FORWARD pwm={pwm_full}, "
          f"SLOW pwm={pwm_slow}, {args.runs} runs each.\n")
    fwd_speeds, slow_speeds, decels = [], [], []
    try:
        for r in range(args.runs):
            print(f"  run {r+1}/{args.runs}: FORWARD…")
            v, coast = _measure_once(motor, ultra, pwm_full, args.drive_seconds, args.min_distance_cm)
            if v > 0:
                fwd_speeds.append(v)
            if v > 0 and coast > 0:
                decels.append(decel_from_coast(v, coast))
            time.sleep(0.5)
            print(f"  run {r+1}/{args.runs}: SLOW…")
            vs, _ = _measure_once(motor, ultra, pwm_slow, args.drive_seconds, args.min_distance_cm)
            if vs > 0:
                slow_speeds.append(vs)
            time.sleep(0.5)
    finally:
        motor.setMotorModel(0, 0)
        motor.close()
        ultra.close()

    if not (fwd_speeds and slow_speeds and decels):
        print("Not enough valid runs (check the wall alignment and ultrasonic).", file=sys.stderr)
        return 1

    values = {
        "forward_speed_mps": float(np.median(fwd_speeds)),
        "slow_speed_mps": float(np.median(slow_speeds)),
        "max_decel_mps2": float(np.median(decels)),
    }
    problems = sanity_problems(**values)
    print("\nMeasured (median of runs):")
    for k in _TARGET_KEYS:
        print(f"  {k}: {values[k]:.3f}")
    if problems:
        print("\nNOT plausible — refusing to emit/apply:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 2

    print("\nPaste into config.yaml:\n")
    print(config_block(values))
    if args.apply:
        backup = patch_config_governor(args.apply, {k: f"{values[k]:.3f}" for k in _TARGET_KEYS})
        print(f"Applied to {args.apply} (backup at {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
