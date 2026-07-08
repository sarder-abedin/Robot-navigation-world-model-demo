"""
run_report.py – offline plots of one navigation run (matplotlib, backend-agnostic).

Reads a run's navigation_log.csv and builds figures for the risk breakdown,
distance, action timeline, latency and network, plus a summary dict. Figures are
built with matplotlib.figure.Figure (no pyplot / no global backend), so the same
functions serve both the PyQt visualizer (Qt canvas) and headless PNG export
(Agg canvas) — and are unit-testable without a display.

    data = load_run(run_dir)
    figs = build_all_figures(data)        # {"risk": Figure, "distance": …}
    save_pngs(run_dir)                     # writes <run>/viz/*.png
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
from matplotlib.figure import Figure

# Action colours (match the HUD / viewers).
ACTION_COLOURS = {
    "FORWARD": "#1a7a1a", "SLOW": "#c8841a", "STOP": "#8b0000",
    "REROUTE": "#7a3a00", "BACKUP": "#a0521a", "TURN": "#1a5a7a",
}
LOW_RISK_MAX, MED_RISK_MAX = 0.25, 0.50   # threshold guide lines


def _col(rows, key, default=math.nan):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, default)))
        except (TypeError, ValueError):
            out.append(default)
    return np.array(out, dtype=float)


def load_run(run_dir: str) -> dict:
    """Parse navigation_log.csv into arrays keyed by column (relative time added)."""
    path = os.path.join(run_dir, "navigation_log.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty log: {path}")
    ts = _col(rows, "timestamp")
    t0 = np.nanmin(ts)
    d = {
        "run_dir": run_dir,
        "cols": set(rows[0].keys()),
        "t": ts - t0,
        "timestamp": ts,
        "frame_idx": _col(rows, "frame_idx"),
        "action": np.array([(r.get("action") or "").strip() for r in rows]),
    }
    for k in ("risk_score", "detector_risk", "world_model_risk", "temporal_risk",
              "ultrasonic_cm", "depth_center_m", "depth_left_m", "depth_right_m",
              "lat_total_ms", "lat_yolo_ms", "lat_wm_ms", "lat_depth_ms",
              "lat_ssv2_ms", "reaction_ema_ms", "net_recv_fps", "net_frames_dropped",
              "net_frames_recv"):
        d[k] = _col(rows, k)
    # distances in metres; -1/blind → nan so gaps show as gaps
    d["ultrasonic_m"] = np.where(d["ultrasonic_cm"] > 0, d["ultrasonic_cm"] / 100.0, np.nan)
    for k in ("depth_center_m", "depth_left_m", "depth_right_m"):
        d[k] = np.where(d[k] > 0, d[k], np.nan)
    return d


# ── Figures ─────────────────────────────────────────────────────────────────

def _has(data, key):
    return key in data and np.isfinite(data[key]).any()


def _legend(ax, **kw):
    """Add a legend only if there are labelled artists (avoids matplotlib warnings)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def figure_risk(data) -> Figure:
    fig = Figure(figsize=(9, 3.4)); ax = fig.add_subplot(111)
    t = data["t"]
    ax.plot(t, data["risk_score"], lw=2.0, color="#111", label="fused risk")
    ax.plot(t, data["detector_risk"], lw=1.0, color="#1a7a1a", label="detector (YOLO)")
    ax.plot(t, data["world_model_risk"], lw=1.0, color="#8a2be2", label="V-JEPA 2")
    ax.plot(t, data["temporal_risk"], lw=1.0, color="#c8841a", label="temporal")
    ax.axhline(LOW_RISK_MAX, ls="--", lw=0.8, color="#888")
    ax.axhline(MED_RISK_MAX, ls="--", lw=0.8, color="#c33")
    ax.set_ylim(0, 1); ax.set_xlabel("time (s)"); ax.set_ylabel("risk")
    ax.set_title("Risk breakdown over time")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def figure_distance(data) -> Figure:
    fig = Figure(figsize=(9, 3.0)); ax = fig.add_subplot(111)
    t = data["t"]
    if _has(data, "ultrasonic_m"):
        ax.plot(t, data["ultrasonic_m"], lw=1.2, color="#c33", label="ultrasonic")
    if _has(data, "depth_center_m"):
        ax.plot(t, data["depth_center_m"], lw=1.2, color="#0a84a8", label="depth centre")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance ahead (m)")
    ax.set_title("Distance to obstacle over time")
    _legend(ax, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def figure_actions(data) -> Figure:
    fig = Figure(figsize=(9, 2.2)); ax = fig.add_subplot(111)
    t, act = data["t"], data["action"]
    # contiguous action segments → coloured bars
    i = 0
    n = len(act)
    seen = []
    while i < n:
        j = i
        while j + 1 < n and act[j + 1] == act[i]:
            j += 1
        start = t[i]
        end = t[j] if j + 1 >= n else t[j + 1]
        colour = ACTION_COLOURS.get(act[i], "#666")
        ax.axvspan(start, max(end, start + 1e-3), color=colour)
        if act[i] not in seen:
            seen.append(act[i])
        i = j + 1
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title("Action timeline")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACTION_COLOURS.get(a, "#666"), label=a) for a in seen],
              loc="upper right", ncol=len(seen) or 1, fontsize=8)
    fig.tight_layout()
    return fig


def figure_latency(data) -> Figure:
    fig = Figure(figsize=(9, 3.0)); ax = fig.add_subplot(111)
    t = data["t"]
    for key, colour, lbl in (("lat_total_ms", "#111", "total"),
                             ("lat_yolo_ms", "#1a7a1a", "yolo"),
                             ("lat_wm_ms", "#8a2be2", "v-jepa2"),
                             ("lat_depth_ms", "#0a84a8", "depth"),
                             ("lat_ssv2_ms", "#c8841a", "ssv2")):
        if _has(data, key):
            ax.plot(t, data[key], lw=(1.8 if key == "lat_total_ms" else 0.9),
                    label=lbl, color=colour)
    ax.set_xlabel("time (s)"); ax.set_ylabel("latency (ms)")
    ax.set_title("Inference latency over time")
    _legend(ax, loc="upper right", ncol=5, fontsize=8)
    fig.tight_layout()
    return fig


def figure_network(data) -> Figure:
    fig = Figure(figsize=(9, 3.0)); ax = fig.add_subplot(111)
    t = data["t"]
    if _has(data, "net_recv_fps"):
        ax.plot(t, data["net_recv_fps"], lw=1.2, color="#0a84a8", label="recv fps")
        ax.set_ylabel("camera fps", color="#0a84a8")
    if _has(data, "net_frames_dropped"):
        ax2 = ax.twinx()
        ax2.plot(t, data["net_frames_dropped"], lw=1.2, color="#c33", label="frames dropped")
        ax2.set_ylabel("cumulative frames dropped", color="#c33")
    ax.set_xlabel("time (s)"); ax.set_title("Network / stream")
    fig.tight_layout()
    return fig


_FIGS = {
    "risk": figure_risk, "distance": figure_distance, "actions": figure_actions,
    "latency": figure_latency, "network": figure_network,
}


def build_all_figures(data) -> dict:
    return {name: fn(data) for name, fn in _FIGS.items()}


def summary(data) -> dict:
    t = data["t"]
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    n = len(t)
    acts, counts = np.unique(data["action"], return_counts=True)
    action_pct = {a: round(100.0 * c / n, 1) for a, c in zip(acts, counts)}
    wm = data["world_model_risk"][np.isfinite(data["world_model_risk"])]
    lat = data["lat_total_ms"][np.isfinite(data["lat_total_ms"])]
    recv = data["net_frames_recv"][np.isfinite(data["net_frames_recv"])]
    drop = data["net_frames_dropped"][np.isfinite(data["net_frames_dropped"])]
    drop_pct = (100.0 * drop[-1] / recv[-1]) if (recv.size and drop.size and recv[-1] > 0) else float("nan")
    return {
        "frames": n,
        "duration_s": round(dur, 1),
        "processed_fps": round(n / dur, 1) if dur > 0 else 0.0,
        "action_pct": action_pct,
        "wm_risk_mean": round(float(np.mean(wm)), 3) if wm.size else None,
        "wm_risk_std": round(float(np.std(wm)), 4) if wm.size else None,
        "lat_total_p50_ms": round(float(np.percentile(lat, 50)), 1) if lat.size else None,
        "lat_total_p95_ms": round(float(np.percentile(lat, 95)), 1) if lat.size else None,
        "frames_dropped_pct": round(drop_pct, 1) if not math.isnan(drop_pct) else None,
    }


def summary_text(data) -> str:
    s = summary(data)
    lines = [
        f"frames        : {s['frames']}",
        f"duration      : {s['duration_s']} s   ({s['processed_fps']} fps processed)",
        f"actions       : " + ", ".join(f"{a} {p}%" for a, p in s["action_pct"].items()),
        f"V-JEPA2 risk  : mean {s['wm_risk_mean']}  std {s['wm_risk_std']}"
        + ("   (flat → uncalibrated anchors)" if s["wm_risk_std"] is not None and s["wm_risk_std"] < 0.03 else ""),
        f"latency total : p50 {s['lat_total_p50_ms']} ms   p95 {s['lat_total_p95_ms']} ms",
        f"frames dropped: {s['frames_dropped_pct']}%",
    ]
    return "\n".join(lines)


def save_pngs(run_dir: str, out_dir: str | None = None, dpi: int = 110) -> list[str]:
    """Save one PNG per chart into <run>/viz/ (default) and a summary.txt."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    data = load_run(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fig in build_all_figures(data).items():
        FigureCanvasAgg(fig)
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    txt = os.path.join(out_dir, "summary.txt")
    with open(txt, "w") as f:
        f.write(summary_text(data) + "\n")
    paths.append(txt)
    return paths
