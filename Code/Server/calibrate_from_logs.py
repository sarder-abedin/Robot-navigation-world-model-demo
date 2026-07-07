"""
calibrate_from_logs.py – calibrate from a stored run, with ZERO extra driving.

Reuses the data a normal run already records in logs_rpi/<run>/:
  • depth.scale        = median(ultrasonic_m / depth_center_m) over frames where the
                         sonar is a valid mid-range reading (the sonar is a free
                         ground-truth ruler).
  • governor speeds    = distance-vs-time slope during FORWARD / SLOW stretches
                         (forward_speed_mps, slow_speed_mps; deceleration from a
                         FORWARD→STOP coast when the logs contain one).
  • V-JEPA 2 anchors    (--anchors) = raw frames auto-labelled blocked/clear from
                         INDEPENDENT signals (YOLO + ultrasonic + action, never the
                         world-model itself) → the same encoder calibrate_anchors uses.

Usage:
  cd Code/Server
  python calibrate_from_logs.py --run ../../logs_rpi/run_YYYYMMDD_HHMMSS_predictive
  python calibrate_from_logs.py --run <dir> --apply config.yaml            # patch depth+governor
  python calibrate_from_logs.py --run <dir> --anchors --apply config.yaml  # also build anchors

Requirements:
  • The ultrasonic must have been WORKING in that run (valid readings) for depth +
    governor. Frames with no echo are skipped; if too few remain the tool says so.
  • --anchors needs raw frames → enable logging.save_raw_frames before the run, and
    run this where V-JEPA 2 loads (GPU/MPS box). depth+governor need only numpy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import shutil
import sys

import numpy as np


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_rows(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "navigation_log.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


# ── 1. Depth scale (sonar = ground truth) ─────────────────────────────────────

def depth_scale_from_rows(rows: list[dict], sonar_lo_cm: float = 15.0,
                          sonar_hi_cm: float = 350.0, min_pairs: int = 20):
    """Return (scale, n_pairs). scale = actual/reported; None if too few pairs."""
    ratios = []
    for r in rows:
        son = _f(r, "ultrasonic_cm")
        dc = _f(r, "depth_center_m")
        if not (sonar_lo_cm <= son <= sonar_hi_cm) or not (dc > 0):
            continue
        ratio = (son / 100.0) / dc
        if 0.2 <= ratio <= 5.0:                 # drop wild mismatches
            ratios.append(ratio)
    if len(ratios) < min_pairs:
        return None, len(ratios)
    return float(np.median(ratios)), len(ratios)


# ── 2. Governor speeds (distance-vs-time during FORWARD/SLOW) ──────────────────

def _speed_from_samples(samples: list[tuple[float, float]]) -> float:
    """Speed (m/s) = -slope of distance-vs-time as the robot approaches the wall."""
    if len(samples) < 3:
        return 0.0
    t = np.array([s[0] for s in samples], dtype=float)
    d = np.array([s[1] for s in samples], dtype=float)
    return max(0.0, -float(np.polyfit(t, d, 1)[0]))


def governor_from_rows(rows: list[dict], sonar_lo_cm: float = 15.0,
                       sonar_hi_cm: float = 350.0, min_advance_m: float = 0.10) -> dict:
    """Estimate forward/slow speed + decel from contiguous action segments."""
    fwd, slow, decels = [], [], []
    seg: list[tuple[float, float]] = []       # (t, d_m) in the current action run
    seg_action = None
    prev_fwd_speed = None

    def flush():
        nonlocal seg, seg_action, prev_fwd_speed
        if seg_action in ("FORWARD", "SLOW") and len(seg) >= 3 and seg[0][1] - seg[-1][1] > min_advance_m:
            v = _speed_from_samples(seg)
            if 0.02 < v < 2.0:
                (fwd if seg_action == "FORWARD" else slow).append(v)
                if seg_action == "FORWARD":
                    prev_fwd_speed = (v, seg[-1][1])   # (speed, last distance)
        seg = []

    for r in rows:
        act = (r.get("action") or "").strip()
        son, t = _f(r, "ultrasonic_cm"), _f(r, "timestamp")
        valid = sonar_lo_cm <= son <= sonar_hi_cm and not math.isnan(t)
        if act != seg_action:
            # A FORWARD stretch that ends in STOP → coast = extra distance closed.
            if seg_action == "FORWARD" and act == "STOP" and prev_fwd_speed and valid:
                v, d_at_stop = prev_fwd_speed
                coast = d_at_stop - son / 100.0
                if 0.0 < coast < 1.0:
                    decels.append((v * v) / (2.0 * coast))
            flush()
            seg_action = act
        if act in ("FORWARD", "SLOW") and valid:
            seg.append((t, son / 100.0))
    flush()

    return {
        "forward_speed_mps": float(np.median(fwd)) if fwd else None,
        "slow_speed_mps": float(np.median(slow)) if slow else None,
        "max_decel_mps2": float(np.median(decels)) if decels else None,
        "n_forward": len(fwd), "n_slow": len(slow), "n_decel": len(decels),
    }


# ── 3. Auto-label frames blocked / clear (independent of the world model) ──────

def autolabel_rows(rows: list[dict]) -> tuple[list[int], list[int]]:
    """Return (blocked_frame_idxs, clear_frame_idxs) from YOLO + sonar + action.

    Deliberately does NOT use world_model_risk (that's what we're calibrating).
    """
    blocked, clear = [], []
    for r in rows:
        try:
            idx = int(_f(r, "frame_idx"))
        except (TypeError, ValueError):
            continue
        act = (r.get("action") or "").strip()
        det, son = _f(r, "detector_risk"), _f(r, "ultrasonic_cm")
        obst, center, risk = _f(r, "obstacles"), _f(r, "in_center"), _f(r, "risk_score")
        son_close = 0 < son <= 45
        son_far = son >= 100
        if son_close or (det >= 0.6 and center >= 1) or (act in ("STOP", "REROUTE", "BACKUP") and det >= 0.4):
            blocked.append(idx)
        elif act == "FORWARD" and risk <= 0.30 and obst < 1 and (son <= 0 or son_far):
            clear.append(idx)
    return blocked, clear


def _balance(a: list, b: list, cap: int = 40) -> tuple[list, list]:
    """Evenly subsample both lists to the same size (≤ cap) for balanced anchors."""
    n = min(len(a), len(b), cap)
    def pick(xs):
        if len(xs) <= n:
            return xs
        step = len(xs) / n
        return [xs[int(i * step)] for i in range(n)]
    return pick(a), pick(b)


# ── Config patching (surgical; preserves comments) ────────────────────────────

def patch_config_block(path: str, block: str, updates: dict) -> str:
    """Patch key: value pairs inside a named YAML block, preserving comments.

    Backs up to <path>.bak and restores on any failure. String values are quoted.
    """
    with open(path) as f:
        lines = f.readlines()
    blk_idx = blk_indent = None
    for i, ln in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(block)}:\s*(#.*)?$", ln)
        if m:
            blk_idx, blk_indent = i, len(m.group(1))
            break
    if blk_idx is None:
        raise ValueError(f"no `{block}:` block found in {path}")
    remaining = dict(updates)
    i = blk_idx + 1
    while i < len(lines) and remaining:
        ln = lines[i]
        if ln.strip() and not ln.lstrip().startswith("#"):
            indent = len(ln) - len(ln.lstrip())
            if indent <= blk_indent:
                break
            m = re.match(r"^(\s*)([A-Za-z0-9_]+):\s*([^#\n]*)(#.*)?$", ln)
            if m and m.group(2) in remaining:
                key = m.group(2)
                val = remaining.pop(key)
                valstr = f'"{val}"' if isinstance(val, str) else f"{val}"
                comment = m.group(4) or ""
                new = f"{m.group(1)}{key}: {valstr}" + (f"  {comment}" if comment else "")
                lines[i] = new.rstrip() + "\n"
        i += 1
    if remaining:
        raise ValueError(f"keys not found in `{block}` block: {sorted(remaining)}")
    backup = path + ".bak"
    shutil.copyfile(path, backup)
    with open(path, "w") as f:
        f.writelines(lines)
    try:
        import yaml
        yaml.safe_load(open(path))          # must still be valid YAML
    except Exception:
        shutil.copyfile(backup, path)
        raise
    return backup


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_anchors_from_run(run_dir: str, rows: list[dict], out: str, config: str) -> bool:
    import cv2
    import yaml
    raw_dir = os.path.join(run_dir, "raw_frames")
    if not os.path.isdir(raw_dir):
        print(f"  anchors: no raw_frames/ in {run_dir} — enable logging.save_raw_frames "
              f"and do one normal run first.", file=sys.stderr)
        return False
    blocked_idx, clear_idx = autolabel_rows(rows)
    blocked_idx, clear_idx = _balance(sorted(set(blocked_idx)), sorted(set(clear_idx)))

    def load(idxs):
        imgs = []
        for idx in idxs:
            p = os.path.join(raw_dir, f"frame_{idx:06d}.jpg")
            bgr = cv2.imread(p)
            if bgr is not None:
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return imgs

    blocked, clear = load(blocked_idx), load(clear_idx)
    if len(blocked) < 3 or len(clear) < 3:
        print(f"  anchors: too few labelled raw frames (blocked={len(blocked)}, "
              f"clear={len(clear)}) — need a run with both blocked and clear stretches.",
              file=sys.stderr)
        return False
    from world_model import WorldModel
    wm = WorldModel(yaml.safe_load(open(config)))
    wm.load()
    wm.build_anchors(blocked, clear)
    wm.save_anchors(out)
    print(f"  anchors: built from {len(blocked)} blocked + {len(clear)} clear raw frames → {out}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate depth/governor/anchors from a stored run")
    ap.add_argument("--run", required=True, help="logs_rpi/<run> directory")
    ap.add_argument("--apply", default="", help="config.yaml to patch in place (optional)")
    ap.add_argument("--anchors", action="store_true", help="also build V-JEPA 2 anchors (needs raw_frames/ + the model)")
    ap.add_argument("--anchors-out", default="anchors.npz")
    args = ap.parse_args(argv)

    rows = read_rows(args.run)
    print(f"Read {len(rows)} logged frames from {args.run}")

    scale, n = depth_scale_from_rows(rows)
    gov = governor_from_rows(rows)

    print("\n── Depth scale ──")
    if scale is None:
        print(f"  insufficient sonar/depth pairs ({n}) — need a run with a working "
              f"ultrasonic and depth enabled.")
    else:
        print(f"  depth.scale = {scale:.3f}   (from {n} sonar/depth pairs)")

    print("\n── Governor ──")
    print(f"  forward_speed_mps = {gov['forward_speed_mps']}  (from {gov['n_forward']} segments)")
    print(f"  slow_speed_mps    = {gov['slow_speed_mps']}  (from {gov['n_slow']} segments)")
    print(f"  max_decel_mps2    = {gov['max_decel_mps2']}  (from {gov['n_decel']} coasts; "
          f"keep the config default if None)")

    if args.apply:
        if scale is not None:
            patch_config_block(args.apply, "depth", {"scale": round(scale, 3)})
            print(f"  patched depth.scale in {args.apply}")
        gov_updates = {k: round(v, 3) for k, v in gov.items()
                       if k in ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2") and v}
        if gov_updates:
            patch_config_block(args.apply, "governor", gov_updates)
            print(f"  patched governor {sorted(gov_updates)} in {args.apply}")

    if args.anchors:
        print("\n── Anchors ──")
        cfg_path = args.apply or "config.yaml"
        if _build_anchors_from_run(args.run, rows, args.anchors_out, cfg_path) and args.apply:
            patch_config_block(args.apply, "world_model", {"anchors_path": args.anchors_out})
            print(f"  patched world_model.anchors_path in {args.apply}")

    print("\nDone." + ("" if args.apply else "  (re-run with --apply <config.yaml> to write these in.)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
