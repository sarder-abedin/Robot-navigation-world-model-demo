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
    # Relative time (s). If the timestamp column is missing/blank in every row,
    # np.nanmin would warn and poison the whole x-axis with NaN — fall back to the
    # frame index so the charts still render.
    if np.isfinite(ts).any():
        rel_t = ts - np.nanmin(ts)
    else:
        rel_t = np.arange(len(rows), dtype=float)
    d = {
        "run_dir": run_dir,
        "cols": set(rows[0].keys()),
        "t": rel_t,
        "timestamp": ts,
        "frame_idx": _col(rows, "frame_idx"),
        "action": np.array([(r.get("action") or "").strip() for r in rows]),
        "wm_label": np.array([(r.get("wm_label") or "").strip().upper() for r in rows]),
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
    n = len(t)
    # Duration from the finite time span (robust to a bad boundary timestamp); the
    # rate is frames / span = (n-1) intervals / span, not n / span.
    finite_t = t[np.isfinite(t)]
    dur = float(np.nanmax(finite_t) - np.nanmin(finite_t)) if finite_t.size > 1 else 0.0
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
        "processed_fps": round((n - 1) / dur, 1) if dur > 0 else 0.0,
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


# ── Analysis (insights + auto-notes) ─────────────────────────────────────────

# Fusion weights — must match decision.py (risk = 0.35·det + 0.45·wm + 0.20·temporal).
RISK_WEIGHTS = {"detector_risk": 0.35, "world_model_risk": 0.45, "temporal_risk": 0.20}
_DRIVER_NAME = {"detector_risk": "detector", "world_model_risk": "world_model",
                "temporal_risk": "temporal"}


def _active_pct(arr) -> float:
    """% of frames where a finite value is > 0 (i.e. the model actually fired)."""
    a = arr[np.isfinite(arr)]
    return round(100.0 * float(np.count_nonzero(a > 0)) / a.size, 1) if a.size else 0.0


def analyze(data) -> dict:
    """Derive higher-level insights + human-readable auto-notes from one run.

    Returns a dict with the risk-driver split, action mix, V-JEPA 2 label
    distribution, model-activity / stream-health numbers, and a `notes` list of
    plain-language findings (uncalibrated anchors, ultrasonic-driven, YOLO quiet,
    no-echo, dropped frames, latency) so a report reads like the manual analysis.
    """
    s = summary(data)
    n = s["frames"]

    # Risk driver split: weight × mean(component), normalized to 100%.
    contrib = {}
    for key, w in RISK_WEIGHTS.items():
        vals = data[key][np.isfinite(data[key])]
        contrib[key] = w * float(np.mean(vals)) if vals.size else 0.0
    tot = sum(contrib.values()) or 1.0
    drivers = {_DRIVER_NAME[k]: round(100.0 * v / tot, 1) for k, v in contrib.items()}
    top_driver = max(drivers, key=drivers.get) if drivers else None

    # V-JEPA 2 label distribution (BLOCKED / MIXED / CLEAR …).
    labels = data.get("wm_label")
    wm_label_pct = {}
    if labels is not None and labels.size:
        named = labels[labels != ""]
        if named.size:
            u, c = np.unique(named, return_counts=True)
            wm_label_pct = {lab: round(100.0 * cc / named.size, 1) for lab, cc in zip(u, c)}

    det_active = _active_pct(data["detector_risk"])
    wm_active = _active_pct(data["world_model_risk"])
    # Ultrasonic no-echo: cm ≤ 0 (the -1 sentinel) among rows that logged the field.
    us = data["ultrasonic_cm"][np.isfinite(data["ultrasonic_cm"])]
    noecho_pct = round(100.0 * float(np.count_nonzero(us <= 0)) / us.size, 1) if us.size else 0.0
    stop_pct = s["action_pct"].get("STOP", 0.0)
    reroute_pct = s["action_pct"].get("REROUTE", 0.0) + s["action_pct"].get("BACKUP", 0.0)

    # The driver line is informational (always shown); `problems` are findings
    # worth acting on. A run with no problems gets the "looks healthy" note.
    notes = []
    if top_driver:
        notes.append(f"Risk is driven mostly by the {top_driver} ({drivers[top_driver]}% of the "
                     f"fused signal).")
    problems = []
    if s["wm_risk_std"] is not None and wm_active > 20 and s["wm_risk_std"] < 0.03:
        problems.append(f"V-JEPA 2 risk is flat (~{s['wm_risk_mean']}) → the anchors look "
                        f"uncalibrated. Calibrating them (calibrate_anchors.py) is the biggest "
                        f"accuracy win.")
    if det_active < 10:
        problems.append(f"YOLO rarely fires ({det_active}% of frames) → mostly plain "
                        f"walls/surfaces it isn't trained on; depth + ultrasonic carry the "
                        f"obstacle sensing.")
    if stop_pct > 25:
        problems.append(f"Heavily ultrasonic-driven (STOP {stop_pct}%) → likely an enclosed or "
                        f"cluttered space with few clearly-open sides.")
    if noecho_pct > 5:
        problems.append(f"Ultrasonic returned no echo on {noecho_pct}% of frames (distance = -1) "
                        f"→ the distance hard-stop was disabled there; check echo wiring/power.")
    if s["frames_dropped_pct"] is not None and s["frames_dropped_pct"] > 40:
        problems.append(f"High dropped-frame rate ({s['frames_dropped_pct']}%) → the network or "
                        f"CPU can't keep up with the camera stream.")
    if s["lat_total_p95_ms"] is not None and s["lat_total_p95_ms"] > 150:
        problems.append(f"p95 latency is {s['lat_total_p95_ms']} ms → the world model or depth "
                        f"stage is the bottleneck.")
    if problems:
        notes += problems
    else:
        notes.append("Run looks healthy: risk well-distributed, models active, stream stable.")

    return {
        "run": os.path.basename(data["run_dir"].rstrip("/")),
        "frames": n,
        "duration_s": s["duration_s"],
        "risk_mean": round(float(np.mean(data["risk_score"][np.isfinite(data["risk_score"])])), 3)
        if np.isfinite(data["risk_score"]).any() else None,
        "risk_drivers_pct": drivers,
        "top_driver": top_driver,
        "action_pct": s["action_pct"],
        "wm_label_pct": wm_label_pct,
        "det_active_pct": det_active,
        "wm_active_pct": wm_active,
        "ultrasonic_noecho_pct": noecho_pct,
        "stop_pct": stop_pct,
        "reroute_pct": round(reroute_pct, 1),
        "frames_dropped_pct": s["frames_dropped_pct"],
        "lat_total_p95_ms": s["lat_total_p95_ms"],
        "notes": notes,
    }


def analysis_text(data) -> str:
    a = analyze(data)
    drivers = ", ".join(f"{k} {v}%" for k, v in a["risk_drivers_pct"].items())
    lines = [
        f"risk drivers  : {drivers}",
        f"models active : YOLO {a['det_active_pct']}%   V-JEPA2 {a['wm_active_pct']}%",
    ]
    if a["wm_label_pct"]:
        lines.append("V-JEPA2 label : " + ", ".join(f"{k} {v}%" for k, v in a["wm_label_pct"].items()))
    lines.append("")
    lines += [f"• {note}" for note in a["notes"]]
    return "\n".join(lines)


# ── Multi-run comparison ─────────────────────────────────────────────────────

_COMPARE_CYCLE = ["#1a7a1a", "#8a2be2", "#c8841a", "#0a84a8", "#c33", "#555",
                  "#1a5a7a", "#a0521a"]


def _run_label(data) -> str:
    return os.path.basename(data["run_dir"].rstrip("/"))


def figure_compare_metric(datas, key, title, ylabel, ylim=None) -> Figure:
    """Overlay one metric (each run on its own relative-time axis) for N runs."""
    fig = Figure(figsize=(9, 3.4)); ax = fig.add_subplot(111)
    for i, d in enumerate(datas):
        if key in d and np.isfinite(d[key]).any():
            ax.plot(d["t"], d[key], lw=1.4, label=_run_label(d),
                    color=_COMPARE_CYCLE[i % len(_COMPARE_CYCLE)])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    _legend(ax, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def figure_action_mix_compare(datas) -> Figure:
    """Grouped stacked bars: the action mix (%) for each run side by side."""
    fig = Figure(figsize=(9, 3.4)); ax = fig.add_subplot(111)
    order = ["FORWARD", "SLOW", "TURN", "REROUTE", "BACKUP", "STOP"]
    labels = [_run_label(d) for d in datas]
    x = np.arange(len(datas))
    bottoms = np.zeros(len(datas))
    for act in order:
        vals = np.array([summary(d)["action_pct"].get(act, 0.0) for d in datas])
        if vals.any():
            ax.bar(x, vals, bottom=bottoms, label=act, color=ACTION_COLOURS.get(act, "#666"))
            bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of frames"); ax.set_title("Action mix by run")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def build_compare_figures(datas) -> dict:
    """Overlaid comparison charts for a list of loaded runs (>= 1)."""
    return {
        "risk": figure_compare_metric(datas, "risk_score", "Fused risk by run", "risk", (0, 1)),
        "distance": figure_compare_metric(datas, "ultrasonic_m",
                                          "Distance ahead (ultrasonic) by run", "distance (m)"),
        "latency": figure_compare_metric(datas, "lat_total_ms",
                                         "Total latency by run", "latency (ms)"),
        "actions": figure_action_mix_compare(datas),
    }


def compare_table(datas) -> list[dict]:
    """One summary+analysis row per run, for a side-by-side table."""
    rows = []
    for d in datas:
        a = analyze(d)
        rows.append({
            "run": a["run"], "frames": a["frames"], "duration_s": a["duration_s"],
            "risk_mean": a["risk_mean"], "top_driver": a["top_driver"],
            "stop_pct": a["stop_pct"], "det_active_pct": a["det_active_pct"],
            "wm_active_pct": a["wm_active_pct"], "noecho_pct": a["ultrasonic_noecho_pct"],
            "dropped_pct": a["frames_dropped_pct"], "lat_p95_ms": a["lat_total_p95_ms"],
        })
    return rows


# ── Shareable HTML report ────────────────────────────────────────────────────

def _fig_to_data_uri(fig, dpi: int = 110) -> str:
    import base64
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _html_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def save_report(run_dirs, out_path: str | None = None) -> str:
    """Build a single self-contained HTML report for one or more runs.

    One run → its five charts + summary + analysis notes. Multiple runs →
    overlaid comparison charts + a per-run table + each run's notes. Images are
    embedded (base64) so the file is shareable on its own. Returns the path.
    """
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    if not run_dirs:
        raise ValueError("no runs given")
    datas = [load_run(r) for r in run_dirs]
    multi = len(datas) > 1

    if multi:
        figs = build_compare_figures(datas)
    else:
        figs = build_all_figures(datas[0])
    imgs = "".join(
        f'<figure><figcaption>{_html_escape(name.capitalize())}</figcaption>'
        f'<img src="{_fig_to_data_uri(fig)}" alt="{_html_escape(name)}"></figure>'
        for name, fig in figs.items()
    )

    if multi:
        rows = compare_table(datas)
        headers = ["run", "frames", "duration_s", "risk_mean", "top_driver", "stop_pct",
                   "det_active_pct", "wm_active_pct", "noecho_pct", "dropped_pct", "lat_p95_ms"]
        thead = "".join(f"<th>{_html_escape(h)}</th>" for h in headers)
        tbody = "".join(
            "<tr>" + "".join(f"<td>{_html_escape(r.get(h))}</td>" for h in headers) + "</tr>"
            for r in rows
        )
        table = f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"
        notes_html = "".join(
            f"<h3>{_html_escape(_run_label(d))}</h3><ul>"
            + "".join(f"<li>{_html_escape(nn)}</li>" for nn in analyze(d)["notes"]) + "</ul>"
            for d in datas
        )
        body = table + "<h2>Findings</h2>" + notes_html
        title = f"Navigation report — {len(datas)} runs"
    else:
        a = analyze(datas[0])
        summ = _html_escape(summary_text(datas[0]))
        notes_html = "<ul>" + "".join(f"<li>{_html_escape(nn)}</li>" for nn in a["notes"]) + "</ul>"
        body = f"<pre>{summ}</pre><h2>Findings</h2>{notes_html}"
        title = f"Navigation report — {a['run']}"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{_html_escape(title)}</title><style>
body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#111;background:#fff}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}} h3{{font-size:14px;margin:14px 0 4px}}
figure{{margin:12px 0}} figcaption{{font-weight:600;font-size:13px;margin-bottom:4px}}
img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:right}} th{{background:#f2f2f2}}
td:first-child,th:first-child{{text-align:left}}
pre{{background:#f7f7f7;padding:10px;border-radius:6px;font-size:13px;white-space:pre-wrap}}
ul{{font-size:13px}}</style></head><body>
<h1>{_html_escape(title)}</h1>
<p style="color:#666;font-size:12px">Runs: {_html_escape(", ".join(_run_label(d) for d in datas))}</p>
<h2>Charts</h2>{imgs}
<h2>Summary &amp; analysis</h2>{body}
</body></html>"""

    if out_path is None:
        base = datas[0]["run_dir"]
        name = "report.html" if not multi else "compare_report.html"
        out_path = os.path.join(base, name)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


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
