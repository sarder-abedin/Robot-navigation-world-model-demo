"""
calibration_ui.py – a separate, desk-only PyQt5 UI for calibration from logs.

Zero extra driving: pick one or more stored run folders, review the derived
depth.scale / governor speeds / anchor label counts, and apply them to config.yaml
(with a verification of what was written). This is a *separate* UI from the
operator viewers (ai_viewer.py / streamlit_viewer.py) — it never drives the robot.

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
    QApplication, QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate_from_logs as cal   # noqa: E402  (path set above)


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
                "⚠ RESTART the server — it reads config.yaml at startup, so the "
                "calibrated values take effect on the next run.",
            ]
            self.finished_ok.emit("\n".join(lines))
        except Exception as exc:
            self.failed.emit(f"Apply failed (config restored from backup): {exc}")


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation Calibration (from logs — no driving)")
        self.resize(760, 720)
        self._worker = None
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Paths
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

        # Run selection — pool one OR MORE runs from anywhere (more runs = more robust)
        runs_box = QGroupBox("1 · Select runs  (tick any number — they're pooled; ✓CSV needed, ✓raw for anchors)")
        rl = QVBoxLayout(runs_box)
        self._run_list = QListWidget()
        rl.addWidget(self._run_list)
        run_btns = QHBoxLayout()
        b_add = QPushButton("+ Add run folder…"); b_add.clicked.connect(self._add_run_folder)
        b_add.setToolTip("Add an individual run folder from anywhere (can be outside the logs dir above).")
        b_clear = QPushButton("Clear list"); b_clear.clicked.connect(self._run_list.clear)
        run_btns.addWidget(b_add); run_btns.addWidget(b_clear); run_btns.addStretch(1)
        rl.addLayout(run_btns)
        root.addWidget(runs_box)

        # Actions
        act = QHBoxLayout()
        self._btn_analyze = QPushButton("2 · Analyze depth + governor")
        self._btn_analyze.clicked.connect(self._analyze)
        act.addWidget(self._btn_analyze)
        self._chk_anchors = QCheckBox("Build V-JEPA 2 anchors on apply (slow; needs raw frames)")
        act.addWidget(self._chk_anchors)
        self._btn_apply = QPushButton("3 · Apply to config")
        self._btn_apply.clicked.connect(self._apply)
        act.addWidget(self._btn_apply)
        root.addLayout(act)

        # Results
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setStyleSheet("font-family: monospace;")
        root.addWidget(self._out)

        self._scan_runs()

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
            return
        added = sum(self._add_run_item(r) for r in run_dirs)
        self._log(f"Scanned {self._logs_edit.text()} — added {added} new run(s) "
                  f"({self._run_list.count()} total in the list).")

    def _add_run_folder(self):
        """Add a single run folder chosen from anywhere (may be outside the logs dir)."""
        p = QFileDialog.getExistingDirectory(self, "Select a run folder", self._logs_edit.text())
        if not p:
            return
        if not os.path.exists(os.path.join(p, "navigation_log.csv")):
            self._log(f"'{os.path.basename(p)}' has no navigation_log.csv — added anyway (disabled).")
        if not self._add_run_item(p):
            self._log(f"'{os.path.basename(p)}' is already in the list.")

    def _selected_runs(self):
        out = []
        for i in range(self._run_list.count()):
            it = self._run_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

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

    def _analyze(self):
        if not self._selected_runs():
            self._log("Select at least one run with a CSV.")
            return
        scale, n, gov, _, selected, raw_runs = self._collect()
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

    def _apply(self):
        if not self._selected_runs():
            self._log("Select at least one run with a CSV.")
            return
        scale, n, gov, rows_by_run, selected, raw_runs = self._collect()
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

    def _on_apply_failed(self, msg):
        self._out.setPlainText(msg)
        self._btn_apply.setEnabled(True)

    def _log(self, msg):
        self._out.append(msg)


def main():
    app = QApplication(sys.argv)
    win = CalibrationWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
