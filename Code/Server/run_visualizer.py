"""
run_visualizer.py – offline PyQt5 UI to visualise (and save) one navigation run.

Select one OR more run folders. One run → its risk / distance / action / latency
/ network plots (embedded matplotlib, zoom/pan) + auto-analysis notes + a synced
annotated-frame scrubber. Several runs → overlaid comparison charts + a per-run
table + each run's findings. "Save PNGs" writes the single-run charts to
<run>/viz/; "Save report (HTML)" writes a self-contained, shareable report.
Desk-only; it never touches the robot.

    python Code/Server/run_visualizer.py

The plotting/summary/analysis logic lives in run_report.py (unit-tested); this
file is the Qt front-end.
"""

import glob
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QPushButton, QScrollArea, QSlider,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_report as rr   # noqa: E402


class RunVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation Run Visualizer (offline)")
        self.resize(1200, 900)
        self.setMinimumSize(900, 600)
        self._data = None        # single loaded run (None in compare mode)
        self._datas = []         # all currently loaded runs (1 or many)
        self._cursors = []       # (canvas, axvline)
        self._frames = []        # (frame_idx, filepath)
        self._frame_times = []   # relative time (s) for each scrubber frame
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Run picker
        top = QGroupBox("Run")
        tl = QHBoxLayout(top)
        left = QVBoxLayout()
        row = QHBoxLayout()
        self._logs_edit = QLineEdit(self._default_logs_dir())
        row.addWidget(QLabel("Logs dir:")); row.addWidget(self._logs_edit)
        b = QPushButton("Browse…"); b.clicked.connect(self._browse); row.addWidget(b)
        bs = QPushButton("Scan"); bs.clicked.connect(self._scan); row.addWidget(bs)
        left.addLayout(row)
        self._run_list = QListWidget()
        # Keep the picker compact so the plots get the vertical space (scrolls for
        # more runs); the charts, not the file list, are the point of this window.
        self._run_list.setMaximumHeight(120)
        # Multi-select: one run → detailed view, several → comparison view.
        self._run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._run_list.itemDoubleClicked.connect(lambda *_: self._load())
        left.addWidget(QLabel("Select one run, or several (Ctrl/Shift) to compare:"))
        left.addWidget(self._run_list)
        tl.addLayout(left, 3)
        rbtn = QVBoxLayout()
        self._btn_load = QPushButton("Load selected"); self._btn_load.clicked.connect(self._load)
        self._btn_save = QPushButton("Save PNGs"); self._btn_save.clicked.connect(self._save)
        self._btn_save.setEnabled(False)
        self._btn_report = QPushButton("Save report (HTML)")
        self._btn_report.clicked.connect(self._save_report)
        self._btn_report.setEnabled(False)
        rbtn.addWidget(self._btn_load); rbtn.addWidget(self._btn_save)
        rbtn.addWidget(self._btn_report); rbtn.addStretch(1)
        tl.addLayout(rbtn, 1)
        root.addWidget(top)

        # Main content in a drag-to-resize splitter: PLOTS on top (given the most
        # room), a compact info + frame-scrubber row below. This stops the summary
        # / analysis text from squeezing the graphs — the old stacked layout left
        # the charts clipped.
        split = QSplitter(Qt.Vertical)

        self._tabs = QTabWidget()
        self._tabs.setMinimumHeight(340)
        split.addWidget(self._tabs)

        bottom = QWidget()
        brow = QHBoxLayout(bottom)
        brow.setContentsMargins(0, 0, 0, 0)

        # Summary + analysis, side-by-side with the scrubber and in a scroll area so
        # long findings scroll instead of stealing height from the plots.
        info = QWidget()
        il = QVBoxLayout(info)
        il.setContentsMargins(6, 6, 6, 6)
        self._summary = QLabel("Load a run to see its plots.")
        self._summary.setStyleSheet("font-family: monospace;")
        self._summary.setWordWrap(True)
        self._summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._analysis = QLabel("")
        self._analysis.setWordWrap(True)
        self._analysis.setStyleSheet("font-family: monospace; color:#0a4a0a;")
        self._analysis.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        il.addWidget(self._summary)
        il.addWidget(self._analysis)
        il.addStretch(1)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setWidget(info)
        info_scroll.setMinimumWidth(300)
        brow.addWidget(info_scroll, 2)

        # Frame scrubber
        scrub = QGroupBox("Frame scrubber (synced cursor on the plots)")
        sl = QVBoxLayout(scrub)
        self._frame_img = QLabel("(no annotated frames in this run)")
        self._frame_img.setAlignment(Qt.AlignCenter)
        self._frame_img.setMinimumHeight(180)
        self._frame_img.setStyleSheet("background:#1a1a1a; color:#777;")
        sl.addWidget(self._frame_img, 1)
        row2 = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_scrub)
        self._time_lbl = QLabel("—")
        row2.addWidget(self._slider); row2.addWidget(self._time_lbl)
        sl.addLayout(row2)
        brow.addWidget(scrub, 3)

        split.addWidget(bottom)
        split.setStretchFactor(0, 3)     # plots get the lion's share
        split.setStretchFactor(1, 2)
        split.setSizes([660, 280])
        root.addWidget(split, 1)

        self._scan()

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _default_logs_dir(self):
        return os.path.normpath(os.path.join(HERE, "..", "..", "logs_rpi"))

    def _browse(self):
        p = QFileDialog.getExistingDirectory(self, "Logs directory", self._logs_edit.text())
        if p:
            self._logs_edit.setText(p); self._scan()

    def _scan(self):
        self._run_list.clear()
        for r in sorted(glob.glob(os.path.join(self._logs_edit.text(), "run_*"))):
            if os.path.exists(os.path.join(r, "navigation_log.csv")):
                self._run_list.addItem(r)

    def _selected_runs(self):
        return [it.text() for it in self._run_list.selectedItems()]

    def _load(self):
        runs = self._selected_runs()
        if not runs:
            self._summary.setText("Select one or more runs in the list first.")
            return
        # Build everything atomically: on any failure, reset to a clean empty
        # state rather than leaving a half-updated window.
        try:
            datas = [rr.load_run(r) for r in runs]
            self._datas = datas
            if len(datas) == 1:
                self._data = datas[0]
                self._summary.setText(rr.summary_text(self._data))
                self._analysis.setText(rr.analysis_text(self._data))
                self._build_plots()
                self._setup_scrubber(runs[0])
            else:
                self._data = None                       # scrubber is single-run only
                self._summary.setText(f"Comparing {len(datas)} runs.")
                self._analysis.setText(
                    "\n".join(f"[{rr._run_label(d)}]  " + rr.analyze(d)["notes"][0] for d in datas))
                self._build_compare_plots()
                self._teardown_scrubber("Select a single run to scrub its frames.")
        except Exception as exc:
            self._data = None; self._datas = []
            self._clear_plots()
            self._teardown_scrubber()
            self._btn_save.setEnabled(False); self._btn_report.setEnabled(False)
            self._summary.setText(f"Failed to load: {exc}")
            self._analysis.setText("")
            return
        self._btn_save.setEnabled(len(datas) == 1)      # PNGs are per single run
        self._btn_report.setEnabled(True)               # report handles 1 or many

    def _clear_plots(self):
        """Remove and delete the previous run's tabs/canvases/figures (no leak)."""
        for i in reversed(range(self._tabs.count())):
            w = self._tabs.widget(i)
            self._tabs.removeTab(i)
            if w is not None:
                w.deleteLater()
        self._cursors = []

    def _add_plot_tab(self, name, fig, cursor=False):
        """One tab = toolbar + the figure in a scroll area, sized to the figure's
        natural height so it's shown FULLY (scrolls if the pane is short) instead
        of being squashed into a clipped strip."""
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        canvas = FigureCanvas(fig)
        # Keep the figure at least its natural pixel height; width flexes with the
        # viewport (no horizontal scroll), height scrolls only when space is tight.
        _, h_in = fig.get_size_inches()
        canvas.setMinimumHeight(int(h_in * fig.get_dpi()))
        pl.addWidget(NavToolbar(canvas, page))
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(canvas)
        pl.addWidget(area, 1)
        if cursor and fig.axes:
            # a dotted vertical cursor on the primary axes, moved by the scrubber
            line = fig.axes[0].axvline(self._data["t"][0], color="k", lw=1.0, ls=":")
            self._cursors.append((canvas, line))
        self._tabs.addTab(page, name.capitalize())

    def _build_plots(self):
        self._clear_plots()
        for name, fig in rr.build_all_figures(self._data).items():
            self._add_plot_tab(name, fig, cursor=True)

    def _build_compare_plots(self):
        """Overlaid comparison charts across the loaded runs (no time cursor)."""
        self._clear_plots()
        for name, fig in rr.build_compare_figures(self._datas).items():
            self._add_plot_tab(name, fig, cursor=False)

    def _teardown_scrubber(self, msg="(no annotated frames in this run)"):
        self._frames = []; self._frame_times = []
        self._slider.setEnabled(False)
        self._frame_img.setText(msg)
        self._time_lbl.setText("—")

    def _setup_scrubber(self, run):
        frames = sorted(glob.glob(os.path.join(run, "frames", "frame_*.jpg")))
        self._frames = []
        self._frame_times = []
        fidx, ftime = self._data["frame_idx"], self._data["t"]
        have_idx = np.isfinite(fidx).any()
        for k, p in enumerate(frames):
            try:
                idx = int(os.path.basename(p).split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if have_idx:
                j = int(np.argmin(np.abs(fidx - idx)))     # nearest logged frame index
            else:
                # Degenerate log with no usable frame_idx → map by position instead
                # of pinning every frame's cursor at t[0].
                j = min(k, len(ftime) - 1)
            self._frames.append(p)
            self._frame_times.append(float(ftime[j]))
        has = bool(self._frames)
        self._slider.setEnabled(has)
        if has:
            self._slider.setMinimum(0); self._slider.setMaximum(len(self._frames) - 1)
            self._slider.setValue(0); self._on_scrub(0)
        else:
            self._frame_img.setText("(no annotated frames saved in this run)")
            self._time_lbl.setText("—")

    def _on_scrub(self, i):
        if not self._frames:
            return
        i = max(0, min(i, len(self._frames) - 1))
        pix = QPixmap(self._frames[i])
        if not pix.isNull():
            # Fit within the (resizable) frame area, preserving aspect ratio.
            w = max(160, self._frame_img.width() - 6)
            h = max(120, self._frame_img.height() - 6)
            self._frame_img.setPixmap(
                pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        t = self._frame_times[i]
        self._time_lbl.setText(f"t = {t:.2f} s   (frame {os.path.basename(self._frames[i])})")
        for canvas, line in self._cursors:
            line.set_xdata([t, t])
            canvas.draw_idle()

    def _save(self):
        # Save the LOADED run (the one shown), not whatever is highlighted in the
        # list — otherwise the saved PNGs wouldn't match the displayed plots/summary.
        if self._data is None:
            self._summary.setText("Load a run before saving.")
            return
        run = self._data["run_dir"]
        try:
            paths = rr.save_pngs(run)
            self._summary.setText(rr.summary_text(self._data)
                                  + f"\n\nSaved {len(paths)} files → {os.path.join(run, 'viz')}")
        except Exception as exc:
            self._summary.setText(f"Save failed: {exc}")

    def _save_report(self):
        """Write a self-contained, shareable HTML report of the loaded run(s)."""
        if not self._datas:
            self._summary.setText("Load run(s) before saving a report.")
            return
        runs = [d["run_dir"] for d in self._datas]
        default = os.path.join(runs[0],
                               "report.html" if len(runs) == 1 else "compare_report.html")
        path, _ = QFileDialog.getSaveFileName(self, "Save report", default,
                                              "HTML files (*.html)")
        if not path:
            return
        try:
            out = rr.save_report(runs, out_path=path)
            self._summary.setText(f"Saved report ({len(runs)} run(s)) → {out}")
        except Exception as exc:
            self._summary.setText(f"Report failed: {exc}")


def main():
    app = QApplication(sys.argv)
    win = RunVisualizer(); win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
