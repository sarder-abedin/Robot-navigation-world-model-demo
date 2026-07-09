"""
calibration_ui.py – a separate, desk-only PyQt5 UI for calibration from logs.

A **step-by-step guided workflow**: each numbered step shows what to do, whether
it is MANDATORY / RECOMMENDED / OPTIONAL, and a live status (done / next /
waiting / not-ready). Zero extra driving — you pick one or more stored run
folders, review the derived depth.scale / governor speeds / anchor labels, and
apply them to config.yaml (with a verification of what was written). This is a
*separate* UI from the operator viewers (ai_viewer.py / streamlit_viewer.py) —
it never drives the robot.

The guidance is a *soft guide*: it always tells you the recommended next step
and flags steps that aren't ready, but it never blocks you from running a step
out of order.

Run:
    python Code/Server/calibration_ui.py

It reuses the tested pure functions in calibrate_from_logs.py, so the numbers match
the CLI exactly. Anchor building (which loads V-JEPA 2) runs in a worker thread so
the window never freezes.
"""

import glob
import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate_from_logs as cal   # noqa: E402  (path set above)

# ── Step requirement levels (badge text + colour) ─────────────────────────────
MANDATORY = ("MANDATORY", "#8b0000")
RECOMMENDED = ("RECOMMENDED", "#c8841a")
OPTIONAL = ("OPTIONAL", "#556")

# ── Step statuses (dot glyph + colour) ────────────────────────────────────────
ST_DONE = ("✓ done", "#1a7a1a")
ST_NEXT = ("→ do this next", "#0a6")
ST_WAITING = ("• waiting", "#888")
ST_NOT_READY = ("✗ not ready", "#b06a00")
ST_INFO = ("• read this", "#556")


class StepBox(QGroupBox):
    """One numbered workflow step: title + level badge + live status + body."""

    def __init__(self, number: int, title: str, level, hint: str):
        super().__init__()
        self.setObjectName("stepBox")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)

        header = QHBoxLayout()
        self._title = QLabel(f"<b>Step {number} · {title}</b>")
        header.addWidget(self._title)
        badge = QLabel(level[0])
        badge.setStyleSheet(
            f"color:white; background:{level[1]}; padding:1px 8px; border-radius:8px; "
            "font-size:10px; font-weight:bold;")
        header.addWidget(badge)
        header.addStretch(1)
        self._status = QLabel()
        header.addWidget(self._status)
        lay.addLayout(header)

        self._hint = QLabel(hint)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#444;")
        lay.addWidget(self._hint)

        self._body = QVBoxLayout()
        lay.addLayout(self._body)

        self.set_status(ST_WAITING)

    def body(self) -> QVBoxLayout:
        return self._body

    def set_status(self, status):
        text, colour = status
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{colour}; font-weight:bold;")

    def set_hint(self, hint: str):
        self._hint.setText(hint)


class ApplyWorker(QThread):
    """Runs patch + (optional) anchor building off the GUI thread."""
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg, scale, gov, anchor_runs, rows_by_run, build_anchors):
        super().__init__()
        self._cfg, self._scale, self._gov = cfg, scale, gov
        self._anchor_runs, self._rows = anchor_runs, rows_by_run
        self._build_anchors = build_anchors

    def run(self):
        try:
            applied = []
            if self._scale is not None:
                cal.patch_config_block(self._cfg, "depth", {"scale": round(self._scale, 3)})
                applied.append(f"depth.scale = {round(self._scale, 3)}")
            gu = {k: round(v, 3) for k, v in self._gov.items()
                  if k in ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2") and v}
            if gu:
                cal.patch_config_block(self._cfg, "governor", gu)
                applied.append(f"governor {gu}")
            if self._build_anchors and self._anchor_runs:
                out = os.path.abspath(os.path.join(HERE, "anchors.npz"))
                ok = cal._build_anchors_from_runs(
                    [(r, self._rows[r]) for r in self._anchor_runs], out, self._cfg)
                if ok:
                    cal.patch_config_block(self._cfg, "world_model", {"anchors_path": out})
                    applied.append(f"anchors_path = {out}")
            # Verify what actually landed.
            c = yaml.safe_load(open(self._cfg))
            ap = (c.get("world_model", {}) or {}).get("anchors_path", "")
            lines = [
                "Applied: " + ("; ".join(applied) if applied else "nothing (no valid values)"),
                "",
                "── Effective in config now ──",
                f"  config file : {os.path.abspath(self._cfg)}",
                f"  depth.scale : {(c.get('depth', {}) or {}).get('scale')}",
                f"  governor    : {(c.get('decision', {}) or {}).get('governor', {})}",
                f"  anchors_path: {ap}"
                + ("  ✓ file present" if ap and os.path.exists(ap)
                   else ("  ✗ FILE MISSING" if ap else "")),
                "",
                "⚠ RESTART the server (Step 7) — it reads config.yaml at startup, so the "
                "calibrated values take effect on the next run.",
            ]
            self.finished_ok.emit("\n".join(lines))
        except Exception as exc:
            self.failed.emit(f"Apply failed (config restored from backup): {exc}")


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation Calibration — guided (from logs, no driving)")
        self.resize(820, 900)
        self._worker = None
        self._analyzed = None      # last _collect() result once Analyze has run
        self._applied = False
        self._refreshing = False   # re-entrancy guard for _refresh_status
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        intro = QLabel(
            "Follow the steps top to bottom. Badges show whether a step is "
            "<b>MANDATORY</b>, <b>RECOMMENDED</b> or <b>OPTIONAL</b>; the status on the "
            "right updates as you go. Guidance is a soft guide — you can still run any "
            "step out of order.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # Scrollable step column (the workflow can be taller than the window).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        root = QVBoxLayout(holder)
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        # Paths (config + logs dir) — needed by several steps.
        paths = QGroupBox("Paths")
        pg = QGridLayout(paths)
        self._cfg_edit = QLineEdit(os.path.join(HERE, "config.yaml"))
        self._logs_edit = QLineEdit(self._default_logs_dir())
        pg.addWidget(QLabel("Config file:"), 0, 0)
        pg.addWidget(self._cfg_edit, 0, 1)
        b1 = QPushButton("Browse…"); b1.clicked.connect(self._browse_cfg); pg.addWidget(b1, 0, 2)
        pg.addWidget(QLabel("Logs directory:"), 1, 0)
        pg.addWidget(self._logs_edit, 1, 1)
        b2 = QPushButton("Browse…"); b2.clicked.connect(self._browse_logs); pg.addWidget(b2, 1, 2)
        b3 = QPushButton("Scan logs dir"); b3.clicked.connect(self._scan_runs); pg.addWidget(b3, 2, 2)
        root.addWidget(paths)

        # ── Step 0 — Record a run (prerequisite) ──────────────────────────────
        self._step0 = StepBox(
            0, "Record a run", MANDATORY,
            "Before calibrating you need at least one normal run recorded to "
            "logs_rpi/ with logging ON and a WORKING ultrasonic (the sonar is the "
            "ground-truth ruler). For anchors (Step 5) also enable "
            "logging.save_raw_frames before that run so raw camera frames are stored. "
            "Nothing here drives the robot — you only pick already-recorded runs.")
        self._step0.set_status(ST_INFO)
        root.addWidget(self._step0)

        # ── Step 1 — Select run(s) ────────────────────────────────────────────
        self._step1 = StepBox(
            1, "Select run(s)", MANDATORY,
            "Tick one or more recorded runs below. A run needs ✓CSV to be usable; "
            "✓raw means it also has raw frames for anchors. Use “Scan logs dir” to "
            "list runs under the logs directory, or “Add run folder…” to add one "
            "from anywhere.")
        self._run_list = QListWidget()
        self._run_list.setMinimumHeight(150)
        self._run_list.itemChanged.connect(lambda *_: self._refresh_status())
        self._step1.body().addWidget(self._run_list)
        run_btns = QHBoxLayout()
        b_add = QPushButton("+ Add run folder…"); b_add.clicked.connect(self._add_run_folder)
        b_add.setToolTip("Add an individual run folder from anywhere (can be outside the logs dir above).")
        b_clear = QPushButton("Clear list"); b_clear.clicked.connect(self._clear_runs)
        run_btns.addWidget(b_add); run_btns.addWidget(b_clear); run_btns.addStretch(1)
        opt = QLabel("Pooling several runs is OPTIONAL but makes the numbers more robust.")
        opt.setStyleSheet("color:#556; font-style:italic;")
        self._step1.body().addLayout(run_btns)
        self._step1.body().addWidget(opt)
        root.addWidget(self._step1)

        # ── Step 2 — Analyze ──────────────────────────────────────────────────
        self._step2 = StepBox(
            2, "Analyze depth + governor", MANDATORY,
            "Read the selected run(s) and derive the calibrated values without "
            "writing anything. Results appear in the panel at the bottom and fill in "
            "the status of Steps 3–5.")
        self._btn_analyze = QPushButton("Analyze")
        self._btn_analyze.clicked.connect(self._analyze)
        self._step2.body().addWidget(self._btn_analyze)
        root.addWidget(self._step2)

        # ── Step 3 — Depth scale ──────────────────────────────────────────────
        self._step3 = StepBox(
            3, "Depth scale", RECOMMENDED,
            "depth.scale = median(sonar / depth) — corrects Depth-Anything's relative "
            "output into metres so the speed governor’s clear-distance is real. Needs "
            "enough valid sonar/depth pairs. Derived by Analyze; written by Apply.")
        self._lbl_depth = QLabel("—"); self._lbl_depth.setStyleSheet("font-family:monospace;")
        self._step3.body().addWidget(self._lbl_depth)
        root.addWidget(self._step3)

        # ── Step 4 — Governor speeds ──────────────────────────────────────────
        self._step4 = StepBox(
            4, "Governor speeds", RECOMMENDED,
            "forward/slow m/s and deceleration measured from distance-vs-time during "
            "FORWARD/SLOW stretches — lets the governor cap speed so the robot can "
            "always stop in the clear distance. Derived by Analyze; written by Apply.")
        self._lbl_gov = QLabel("—"); self._lbl_gov.setStyleSheet("font-family:monospace;")
        self._step4.body().addWidget(self._lbl_gov)
        root.addWidget(self._step4)

        # ── Step 5 — V-JEPA 2 anchors ─────────────────────────────────────────
        self._step5 = StepBox(
            5, "V-JEPA 2 anchors", RECOMMENDED,
            "Auto-label raw frames blocked/clear (from YOLO + sonar + action, never "
            "the world model) and build the anchor prototypes so V-JEPA 2 gives a real "
            "risk instead of a flat ~0.49. Needs run(s) with ✓raw frames. Built during "
            "Apply when this is ticked (slow — loads V-JEPA 2).")
        self._chk_anchors = QCheckBox("Build V-JEPA 2 anchors on Apply")
        self._chk_anchors.stateChanged.connect(lambda *_: self._refresh_status())
        self._step5.body().addWidget(self._chk_anchors)
        root.addWidget(self._step5)

        # ── Step 6 — Apply to config ──────────────────────────────────────────
        self._step6 = StepBox(
            6, "Apply to config", MANDATORY,
            "Write the derived values into config.yaml (surgically — comments kept, a "
            ".bak backup is made and restored on any error), then verify what landed. "
            "Nothing takes effect until this step.")
        self._btn_apply = QPushButton("Apply to config")
        self._btn_apply.clicked.connect(self._apply)
        self._step6.body().addWidget(self._btn_apply)
        root.addWidget(self._step6)

        # ── Step 7 — Restart the server ───────────────────────────────────────
        self._step7 = StepBox(
            7, "Restart the server", MANDATORY,
            "The server reads config.yaml only at startup, so restart main_server.py "
            "for the calibrated depth scale, governor speeds and anchors to take effect.")
        self._step7.set_status(ST_INFO)
        root.addWidget(self._step7)

        root.addStretch(1)

        # Results / log panel (persistent, below the scroll area).
        line = QFrame(); line.setFrameShape(QFrame.HLine); outer.addWidget(line)
        outer.addWidget(QLabel("Details / results:"))
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setMaximumHeight(180)
        self._out.setStyleSheet("font-family: monospace;")
        outer.addWidget(self._out)

        self._scan_runs()
        self._refresh_status()

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _default_logs_dir(self):
        try:
            c = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
            d = (c.get("logging", {}) or {}).get("log_dir", "../../logs_rpi")
            return os.path.normpath(os.path.join(HERE, d))
        except Exception:
            return os.path.normpath(os.path.join(HERE, "..", "..", "logs_rpi"))

    def _browse_cfg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select config.yaml", HERE, "YAML (*.yaml *.yml)")
        if p:
            self._cfg_edit.setText(p)

    def _browse_logs(self):
        p = QFileDialog.getExistingDirectory(self, "Select logs directory", self._logs_edit.text())
        if p:
            self._logs_edit.setText(p)
            self._scan_runs()

    def _existing_paths(self) -> set:
        return {self._run_list.item(i).data(Qt.UserRole) for i in range(self._run_list.count())}

    def _add_run_item(self, run_dir: str, check: bool = True) -> bool:
        """Add one run to the list (deduped by path). Returns True if added."""
        run_dir = os.path.normpath(run_dir)
        if run_dir in self._existing_paths():
            return False
        has_csv = os.path.exists(os.path.join(run_dir, "navigation_log.csv"))
        has_raw = os.path.isdir(os.path.join(run_dir, "raw_frames"))
        tag = ("✓CSV" if has_csv else "✗CSV") + ("  ✓raw" if has_raw else "  ✗raw")
        it = QListWidgetItem(f"{os.path.basename(run_dir)}    [{tag}]")
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if (has_csv and check) else Qt.Unchecked)
        if not has_csv:
            it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
        it.setToolTip(run_dir)
        it.setData(Qt.UserRole, run_dir)
        self._run_list.addItem(it)
        return True

    def _scan_runs(self):
        """Append (not replace) the run_* folders under the logs dir; dedupes."""
        run_dirs = sorted(glob.glob(os.path.join(self._logs_edit.text(), "run_*")))
        if not run_dirs:
            self._log(f"No run_* folders in {self._logs_edit.text()}")
            self._refresh_status()
            return
        added = sum(self._add_run_item(r) for r in run_dirs)
        self._log(f"Scanned {self._logs_edit.text()} — added {added} new run(s) "
                  f"({self._run_list.count()} total in the list).")
        self._refresh_status()

    def _clear_runs(self):
        self._run_list.clear()
        self._refresh_status()

    def _add_run_folder(self):
        """Add a single run folder chosen from anywhere (may be outside the logs dir)."""
        p = QFileDialog.getExistingDirectory(self, "Select a run folder", self._logs_edit.text())
        if not p:
            return
        if not os.path.exists(os.path.join(p, "navigation_log.csv")):
            self._log(f"'{os.path.basename(p)}' has no navigation_log.csv — added anyway (disabled).")
        if not self._add_run_item(p):
            self._log(f"'{os.path.basename(p)}' is already in the list.")
        self._refresh_status()

    def _selected_runs(self):
        out = []
        for i in range(self._run_list.count()):
            it = self._run_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _selected_raw_runs(self):
        return [r for r in self._selected_runs()
                if os.path.isdir(os.path.join(r, "raw_frames"))]

    def _collect(self):
        """Pool selected runs → (scale, n, gov, rows_by_run, selected, raw_runs)."""
        selected = self._selected_runs()
        rows_by_run, ratios = {}, []
        gpool = {"forward": [], "slow": [], "decel": []}
        for r in selected:
            rows = cal.read_rows(r)
            rows_by_run[r] = rows
            ratios += cal.depth_ratios(rows)
            s = cal.governor_samples(rows)
            for k in gpool:
                gpool[k] += s[k]
        scale, n = cal.depth_scale_from_ratios(ratios)
        gov = cal.summarize_governor(gpool)
        raw_runs = [r for r in selected if os.path.isdir(os.path.join(r, "raw_frames"))]
        return scale, n, gov, rows_by_run, selected, raw_runs

    # ── Step actions ─────────────────────────────────────────────────────────
    def _analyze(self):
        if not self._selected_runs():
            self._log("Step 1 first: select at least one run with a CSV.")
            self._refresh_status()
            return
        scale, n, gov, _, selected, raw_runs = self._collect()
        self._analyzed = (scale, n, gov, selected, raw_runs)
        lines = [f"Pooled across {len(selected)} run(s):", ""]
        lines.append(f"  depth.scale       = {scale:.3f}   ({n} sonar/depth pairs)"
                     if scale is not None else
                     f"  depth.scale       = insufficient ({n} pairs) — need a working ultrasonic")
        lines.append(f"  forward_speed_mps = {gov['forward_speed_mps']}   ({gov['n_forward']} segments)")
        lines.append(f"  slow_speed_mps    = {gov['slow_speed_mps']}   ({gov['n_slow']} segments)")
        lines.append(f"  max_decel_mps2    = {gov['max_decel_mps2']}   ({gov['n_decel']} coasts; "
                     f"keeps config default if None)")
        lines.append(f"  raw_frames runs   = {len(raw_runs)}/{len(selected)} (for anchors)")
        self._out.setPlainText("\n".join(lines))
        self._refresh_status()

    def _apply(self):
        if not self._selected_runs():
            self._log("Step 1 first: select at least one run with a CSV.")
            self._refresh_status()
            return
        scale, n, gov, rows_by_run, selected, raw_runs = self._collect()
        self._analyzed = (scale, n, gov, selected, raw_runs)
        self._btn_apply.setEnabled(False)
        self._log("Applying…" + (" building anchors (loading V-JEPA 2)…" if self._chk_anchors.isChecked() else ""))
        self._worker = ApplyWorker(self._cfg_edit.text(), scale, gov, raw_runs, rows_by_run,
                                   self._chk_anchors.isChecked())
        self._worker.finished_ok.connect(self._on_applied)
        self._worker.failed.connect(self._on_apply_failed)
        self._worker.start()

    def _on_applied(self, msg):
        self._out.setPlainText(msg)
        self._btn_apply.setEnabled(True)
        self._applied = True
        self._refresh_status()

    def _on_apply_failed(self, msg):
        self._out.setPlainText(msg)
        self._btn_apply.setEnabled(True)
        self._refresh_status()

    # ── Live status / guidance (soft guide — never blocks) ────────────────────
    def _refresh_status(self):
        # Mutating _chk_anchors below re-emits stateChanged → _refresh_status;
        # guard so the update runs exactly once per trigger.
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._refresh_status_inner()
        finally:
            self._refreshing = False

    def _refresh_status_inner(self):
        n_sel = len(self._selected_runs())
        n_raw = len(self._selected_raw_runs())

        # Step 1 — selection.
        if n_sel:
            self._step1.set_status(ST_DONE)
        else:
            self._step1.set_status(ST_NEXT)

        # Step 2 — analyze (next once a run is picked, done once analyzed).
        if self._analyzed is not None:
            self._step2.set_status(ST_DONE)
        elif n_sel:
            self._step2.set_status(ST_NEXT)
        else:
            self._step2.set_status(ST_WAITING)

        scale = gov = None
        if self._analyzed is not None:
            scale, _n, gov, _sel, _raw = self._analyzed

        # Step 3 — depth scale readout.
        if self._analyzed is None:
            self._lbl_depth.setText("—  (run Analyze)")
            self._step3.set_status(ST_WAITING)
        elif scale is not None:
            self._lbl_depth.setText(f"depth.scale = {scale:.3f}")
            self._step3.set_status(ST_DONE if self._applied else ST_NEXT)
        else:
            self._lbl_depth.setText("insufficient sonar/depth pairs — need a working-ultrasonic run")
            self._step3.set_status(ST_NOT_READY)

        # Step 4 — governor readout.
        if self._analyzed is None:
            self._lbl_gov.setText("—  (run Analyze)")
            self._step4.set_status(ST_WAITING)
        else:
            self._lbl_gov.setText(
                f"forward={gov['forward_speed_mps']}  slow={gov['slow_speed_mps']}  "
                f"decel={gov['max_decel_mps2']}")
            any_gov = any(gov.get(k) for k in ("forward_speed_mps", "slow_speed_mps", "max_decel_mps2"))
            if any_gov:
                self._step4.set_status(ST_DONE if self._applied else ST_NEXT)
            else:
                self._step4.set_status(ST_NOT_READY)

        # Step 5 — anchors (needs raw frames).
        if n_raw == 0:
            self._chk_anchors.setChecked(False)
            self._chk_anchors.setEnabled(False)
            self._chk_anchors.setText("Build V-JEPA 2 anchors on Apply  (no ✓raw run selected)")
            self._step5.set_status(ST_NOT_READY)
        else:
            self._chk_anchors.setEnabled(True)
            self._chk_anchors.setText(f"Build V-JEPA 2 anchors on Apply  ({n_raw} ✓raw run(s) selected)")
            if self._applied and self._chk_anchors.isChecked():
                self._step5.set_status(ST_DONE)
            elif self._chk_anchors.isChecked():
                self._step5.set_status(ST_NEXT)
            else:
                self._step5.set_status(ST_WAITING)

        # Step 6 — apply.
        if self._applied:
            self._step6.set_status(ST_DONE)
        elif self._analyzed is not None:
            self._step6.set_status(ST_NEXT)
        else:
            self._step6.set_status(ST_WAITING)

        # Step 7 — restart (highlighted once applied).
        self._step7.set_status(ST_NEXT if self._applied else ST_INFO)

    def _log(self, msg):
        self._out.append(msg)


def main():
    app = QApplication(sys.argv)
    win = CalibrationWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
