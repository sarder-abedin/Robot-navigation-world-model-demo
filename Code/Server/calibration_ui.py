"""
calibration_ui.py – a separate, desk-only Streamlit UI for calibration from logs.

Zero extra driving: pick one or more stored run folders, review the derived
depth.scale / governor speeds / anchor label counts, and apply them to config.yaml
(with a verification of what was written). This is a *separate* UI from the
operator viewers (streamlit_viewer.py / ai_viewer.py) — it never drives the robot.

Run:
    streamlit run Code/Server/calibration_ui.py

It reuses the tested pure functions in calibrate_from_logs.py, so the numbers match
the CLI exactly.
"""

import glob
import os
import sys

import streamlit as st
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate_from_logs as cal   # noqa: E402  (path set above)

st.set_page_config(page_title="Navigation Calibration", layout="centered")
st.title("🎛️ Navigation Calibration")
st.caption("Zero-driving calibration from stored run logs. Pick runs → review → apply. "
           "The robot is never driven from here.")


# ── Paths ─────────────────────────────────────────────────────────────────────

cfg_path = st.text_input("Config file to calibrate", os.path.join(HERE, "config.yaml"))


def _default_logs_dir() -> str:
    try:
        c = yaml.safe_load(open(cfg_path))
        d = (c.get("logging", {}) or {}).get("log_dir", "../../logs_rpi")
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), d))
    except Exception:
        return os.path.normpath(os.path.join(HERE, "..", "..", "logs_rpi"))


log_dir = st.text_input("Logs directory", _default_logs_dir())
run_dirs = sorted(glob.glob(os.path.join(log_dir, "run_*")))


# ── 1. Select runs ────────────────────────────────────────────────────────────

st.subheader("1 · Select runs  (more runs = more robust)")
if not run_dirs:
    st.warning(f"No `run_*` folders found in {log_dir}. Record a run with logging on first.")
    st.stop()

selected = []
for r in run_dirs:
    has_csv = os.path.exists(os.path.join(r, "navigation_log.csv"))
    has_raw = os.path.isdir(os.path.join(r, "raw_frames"))
    tag = ("✓CSV" if has_csv else "✗CSV") + ("  ✓raw" if has_raw else "  ✗raw")
    if st.checkbox(f"{os.path.basename(r)}   [{tag}]", value=has_csv,
                   disabled=not has_csv, key=r):
        selected.append(r)

if not selected:
    st.info("Select at least one run that has a `navigation_log.csv`.")
    st.stop()


# ── Pool + compute (fast; numpy only) ─────────────────────────────────────────

runs, rows_by_run = [], {}
ratios = []
gpool = {"forward": [], "slow": [], "decel": []}
for r in selected:
    rows = cal.read_rows(r)
    rows_by_run[r] = rows
    runs.append((r, rows))
    ratios += cal.depth_ratios(rows)
    s = cal.governor_samples(rows)
    for k in gpool:
        gpool[k] += s[k]

scale, n_pairs = cal.depth_scale_from_ratios(ratios)
gov = cal.summarize_governor(gpool)


# ── 2. Results ────────────────────────────────────────────────────────────────

st.subheader("2 · Results (pooled across selected runs)")
c1, c2, c3 = st.columns(3)
c1.metric("depth.scale", f"{scale:.3f}" if scale is not None else "—", f"{n_pairs} pairs")
c2.metric("forward m/s",
          f"{gov['forward_speed_mps']:.3f}" if gov["forward_speed_mps"] else "—",
          f"{gov['n_forward']} segments")
c3.metric("slow m/s",
          f"{gov['slow_speed_mps']:.3f}" if gov["slow_speed_mps"] else "—",
          f"{gov['n_slow']} segments")
st.write(f"**max_decel_mps2:** "
         + (f"{gov['max_decel_mps2']:.3f} (from {gov['n_decel']} coasts)"
            if gov["max_decel_mps2"] else "not measurable from these logs — the config default is kept"))

if scale is None:
    st.warning("Not enough valid ultrasonic/depth pairs — pick runs recorded with a "
               "**working ultrasonic** and depth enabled.")


# ── 3. Anchors (optional, heavy) ──────────────────────────────────────────────

st.subheader("3 · V-JEPA 2 anchors (optional)")
raw_runs = [r for r in selected if os.path.isdir(os.path.join(r, "raw_frames"))]
if not raw_runs:
    st.info("None of the selected runs have `raw_frames/`. Turn on "
            "`logging.save_raw_frames` and record a run to enable anchor calibration.")
do_anchors = st.checkbox("Build anchors from raw frames (slow — loads V-JEPA 2)",
                         value=False, disabled=not raw_runs)


# ── 4. Apply ──────────────────────────────────────────────────────────────────

st.subheader("4 · Apply to config")
if st.button("Apply to config.yaml", type="primary", disabled=not os.path.exists(cfg_path)):
    applied = []
    try:
        if scale is not None:
            cal.patch_config_block(cfg_path, "depth", {"scale": round(scale, 3)})
            applied.append(f"depth.scale = {round(scale, 3)}")
        gov_updates = {k: round(v, 3) for k, v in gov.items()
                       if k in ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2") and v}
        if gov_updates:
            cal.patch_config_block(cfg_path, "governor", gov_updates)
            applied.append(f"governor {gov_updates}")
        if do_anchors and raw_runs:
            with st.spinner("Building anchors (loading V-JEPA 2 — this can take a minute)…"):
                out = os.path.abspath(os.path.join(HERE, "anchors.npz"))
                ok = cal._build_anchors_from_runs(
                    [(r, rows_by_run[r]) for r in raw_runs], out, cfg_path)
            if ok:
                cal.patch_config_block(cfg_path, "world_model", {"anchors_path": out})
                applied.append(f"anchors_path = {out}")
        st.success("Applied: " + ("; ".join(applied) if applied else "nothing (no valid values)"))
    except Exception as exc:
        st.error(f"Apply failed (config restored from backup): {exc}")

    # Verify what actually landed in the file.
    try:
        c = yaml.safe_load(open(cfg_path))
        ap = (c.get("world_model", {}) or {}).get("anchors_path", "")
        st.info("Effective in config now:")
        st.json({
            "config file": os.path.abspath(cfg_path),
            "depth.scale": (c.get("depth", {}) or {}).get("scale"),
            "decision.governor": (c.get("decision", {}) or {}).get("governor", {}),
            "world_model.anchors_path": ap,
            "anchors file present": bool(ap) and os.path.exists(ap),
        })
    except Exception as exc:
        st.error(f"Could not re-read config: {exc}")
    st.warning("⚠ **Restart the server** — it reads `config.yaml` at startup, so the "
               "calibrated values take effect on the next run.")
