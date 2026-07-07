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

def depth_ratios(rows: list[dict], sonar_lo_cm: float = 15.0,
                 sonar_hi_cm: float = 350.0) -> list[float]:
    """Per-frame ultrasonic/depth ratios (the raw samples; pool these across runs)."""
    out = []
    for r in rows:
        son = _f(r, "ultrasonic_cm")
        dc = _f(r, "depth_center_m")
        if not (sonar_lo_cm <= son <= sonar_hi_cm) or not (dc > 0):
            continue
        ratio = (son / 100.0) / dc
        if 0.2 <= ratio <= 5.0:                 # drop wild mismatches
            out.append(ratio)
    return out


def depth_scale_from_ratios(ratios: list[float], min_pairs: int = 20):
    """Return (scale, n_pairs). scale = actual/reported; None if too few pairs."""
    if len(ratios) < min_pairs:
        return None, len(ratios)
    return float(np.median(ratios)), len(ratios)


def depth_scale_from_rows(rows: list[dict], **kw):
    """Single-run convenience wrapper (kept for tests)."""
    min_pairs = kw.pop("min_pairs", 20)
    return depth_scale_from_ratios(depth_ratios(rows, **kw), min_pairs)


# ── 2. Governor speeds (distance-vs-time during FORWARD/SLOW) ──────────────────

def _speed_from_samples(samples: list[tuple[float, float]]) -> float:
    """Speed (m/s) = -slope of distance-vs-time as the robot approaches the wall."""
    if len(samples) < 3:
        return 0.0
    t = np.array([s[0] for s in samples], dtype=float)
    d = np.array([s[1] for s in samples], dtype=float)
    return max(0.0, -float(np.polyfit(t, d, 1)[0]))


def governor_samples(rows: list[dict], sonar_lo_cm: float = 15.0,
                     sonar_hi_cm: float = 350.0, min_advance_m: float = 0.10) -> dict:
    """Collect per-segment speed/decel samples from ONE run's (time-ordered) rows.

    Returns {"forward":[...], "slow":[...], "decel":[...]} — pool these across runs,
    then summarize. Segments never cross a run boundary (call once per run).
    """
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
    return {"forward": fwd, "slow": slow, "decel": decels}


def summarize_governor(samples: dict) -> dict:
    """Median the pooled per-segment samples into governor constants."""
    fwd, slow, decels = samples["forward"], samples["slow"], samples["decel"]
    return {
        "forward_speed_mps": float(np.median(fwd)) if fwd else None,
        "slow_speed_mps": float(np.median(slow)) if slow else None,
        "max_decel_mps2": float(np.median(decels)) if decels else None,
        "n_forward": len(fwd), "n_slow": len(slow), "n_decel": len(decels),
    }


def governor_from_rows(rows: list[dict], **kw) -> dict:
    """Single-run convenience wrapper (kept for tests)."""
    return summarize_governor(governor_samples(rows, **kw))


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

def _build_anchors_from_runs(runs: list[tuple[str, list[dict]]], out: str, config: str) -> bool:
    """Pool auto-labelled raw frames from ALL runs, then build anchors once."""
    import cv2
    import yaml
    blocked_imgs, clear_imgs = [], []

    def load(raw_dir, idxs):
        imgs = []
        for idx in idxs:
            bgr = cv2.imread(os.path.join(raw_dir, f"frame_{idx:06d}.jpg"))
            if bgr is not None:
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return imgs

    for run_dir, rows in runs:
        raw_dir = os.path.join(run_dir, "raw_frames")
        if not os.path.isdir(raw_dir):
            print(f"  anchors: {run_dir} has no raw_frames/ — skipped "
                  f"(enable logging.save_raw_frames for that run).", file=sys.stderr)
            continue
        b_idx, c_idx = autolabel_rows(rows)
        blocked_imgs += load(raw_dir, sorted(set(b_idx)))
        clear_imgs += load(raw_dir, sorted(set(c_idx)))

    # Balance the pooled classes so neither anchor is dominated by one run.
    blocked, clear = _balance(blocked_imgs, clear_imgs, cap=80)
    if len(blocked) < 3 or len(clear) < 3:
        print(f"  anchors: too few labelled raw frames across runs (blocked={len(blocked)}, "
              f"clear={len(clear)}) — need runs with both blocked and clear stretches.",
              file=sys.stderr)
        return False
    from world_model import WorldModel
    wm = WorldModel(yaml.safe_load(open(config)))
    wm.load()
    wm.build_anchors(blocked, clear)
    wm.save_anchors(out)
    print(f"  anchors: built from {len(blocked)} blocked + {len(clear)} clear raw frames "
          f"(pooled across {len(runs)} run(s)) → {out}")
    return True


def _verify_applied(config_path: str) -> None:
    """Re-read the patched config and print the effective calibrated values, so it's
    obvious they'll be picked up on the next run."""
    import yaml
    abspath = os.path.abspath(config_path)
    cfg = yaml.safe_load(open(config_path))
    scale = (cfg.get("depth", {}) or {}).get("scale")
    gov = (cfg.get("decision", {}) or {}).get("governor", {}) or {}
    ap = (cfg.get("world_model", {}) or {}).get("anchors_path", "")
    print("\n── Written to config (effective on next server start) ──")
    print(f"  config file : {abspath}")
    print(f"  depth.scale : {scale}")
    print(f"  governor    : forward={gov.get('forward_speed_mps')} "
          f"slow={gov.get('slow_speed_mps')} decel={gov.get('max_decel_mps2')}")
    print(f"  anchors_path: {ap}" + ("  ✓ file present" if ap and os.path.exists(ap)
                                      else ("  ✗ FILE MISSING" if ap else "")))
    print("  → RESTART the server (it reads config.yaml at startup) to load these.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate depth/governor/anchors from stored run(s)")
    ap.add_argument("--run", nargs="+", required=True,
                    help="one or more logs_rpi/<run> directories (pooled for robustness)")
    ap.add_argument("--apply", default="", help="config.yaml to patch in place (optional)")
    ap.add_argument("--anchors", action="store_true",
                    help="also build V-JEPA 2 anchors (needs raw_frames/ + the model)")
    ap.add_argument("--anchors-out", default="anchors.npz")
    args = ap.parse_args(argv)

    # Read every run; pool depth ratios and governor samples across all of them.
    runs = []
    all_ratios = []
    gov_pool = {"forward": [], "slow": [], "decel": []}
    for run_dir in args.run:
        rows = read_rows(run_dir)
        runs.append((run_dir, rows))
        r = depth_ratios(rows)
        s = governor_samples(rows)
        all_ratios += r
        for k in gov_pool:
            gov_pool[k] += s[k]
        print(f"Read {len(rows):>5} frames from {run_dir}  "
              f"(depth pairs={len(r)}, fwd seg={len(s['forward'])}, slow seg={len(s['slow'])})")
    print(f"→ pooled across {len(runs)} run(s): {len(all_ratios)} depth pairs, "
          f"{len(gov_pool['forward'])} FORWARD + {len(gov_pool['slow'])} SLOW segments")

    scale, n = depth_scale_from_ratios(all_ratios)
    gov = summarize_governor(gov_pool)

    print("\n── Depth scale ──")
    print(f"  depth.scale = {scale:.3f}   (from {n} pooled sonar/depth pairs)" if scale is not None
          else f"  insufficient sonar/depth pairs ({n}) across runs — need working-ultrasonic runs.")

    print("\n── Governor ──")
    print(f"  forward_speed_mps = {gov['forward_speed_mps']}  (from {gov['n_forward']} segments)")
    print(f"  slow_speed_mps    = {gov['slow_speed_mps']}  (from {gov['n_slow']} segments)")
    print(f"  max_decel_mps2    = {gov['max_decel_mps2']}  (from {gov['n_decel']} coasts; "
          f"keep the config default if None)")

    if args.apply:
        if scale is not None:
            patch_config_block(args.apply, "depth", {"scale": round(scale, 3)})
            print(f"  patched depth.scale")
        gov_updates = {k: round(v, 3) for k, v in gov.items()
                       if k in ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2") and v}
        if gov_updates:
            patch_config_block(args.apply, "governor", gov_updates)
            print(f"  patched governor {sorted(gov_updates)}")

    if args.anchors:
        print("\n── Anchors ──")
        cfg_path = args.apply or "config.yaml"
        # Absolute path so the server finds the anchors regardless of its working dir.
        anchors_out = os.path.abspath(args.anchors_out)
        if _build_anchors_from_runs(runs, anchors_out, cfg_path) and args.apply:
            patch_config_block(args.apply, "world_model", {"anchors_path": anchors_out})
            print(f"  patched world_model.anchors_path")

    if args.apply:
        _verify_applied(args.apply)
    else:
        print("\nDone.  (re-run with --apply <config.yaml> to write these in.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
