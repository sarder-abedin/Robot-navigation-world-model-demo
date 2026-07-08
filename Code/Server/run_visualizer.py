"""
run_visualizer.py – offline PyQt5 UI to visualise (and save) one navigation run.

Pick a run folder → see the risk / distance / action / latency / network plots
(embedded matplotlib, zoom/pan) → scrub the annotated frames with a synced time
cursor on the charts → "Save PNGs" writes one image per chart (+ summary.txt) to
<run>/viz/. Desk-only; it never touches the robot.

    python Code/Server/run_visualizer.py

The plotting/summary logic lives in run_report.py (unit-tested); this file is the
Qt front-end.
"""

import glob
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget,
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
        self.resize(1080, 860)
        self._data = None
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
        self._run_list.itemDoubleClicked.connect(lambda *_: self._load())
        left.addWidget(self._run_list)
        tl.addLayout(left, 3)
        rbtn = QVBoxLayout()
        self._btn_load = QPushButton("Load selected"); self._btn_load.clicked.connect(self._load)
        self._btn_save = QPushButton("Save PNGs"); self._btn_save.clicked.connect(self._save)
        self._btn_save.setEnabled(False)
        rbtn.addWidget(self._btn_load); rbtn.addWidget(self._btn_save); rbtn.addStretch(1)
        tl.addLayout(rbtn, 1)
        root.addWidget(top)

        # Plots (tabs) + summary
        self._summary = QLabel("Load a run to see its plots.")
        self._summary.setStyleSheet("font-family: monospace;")
        root.addWidget(self._summary)
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 5)

        # Frame scrubber
        scrub = QGroupBox("Frame scrubber (synced cursor on the plots)")
        sl = QVBoxLayout(scrub)
        self._frame_img = QLabel("(no annotated frames in this run)")
        self._frame_img.setAlignment(Qt.AlignCenter)
        self._frame_img.setFixedHeight(240)
        self._frame_img.setStyleSheet("background:#1a1a1a; color:#777;")
        sl.addWidget(self._frame_img)
        row2 = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_scrub)
        self._time_lbl = QLabel("—")
        row2.addWidget(self._slider); row2.addWidget(self._time_lbl)
        sl.addLayout(row2)
        root.addWidget(scrub, 3)

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

    def _selected_run(self):
        it = self._run_list.currentItem()
        return it.text() if it else None

    def _load(self):
        run = self._selected_run()
        if not run:
            self._summary.setText("Select a run in the list first.")
            return
        try:
            self._data = rr.load_run(run)
        except Exception as exc:
            self._summary.setText(f"Failed to load: {exc}")
            return
        self._summary.setText(rr.summary_text(self._data))
        self._build_plots()
        self._setup_scrubber(run)
        self._btn_save.setEnabled(True)

    def _build_plots(self):
        self._tabs.clear()
        self._cursors = []
        for name, fig in rr.build_all_figures(self._data).items():
            page = QWidget(); pl = QVBoxLayout(page)
            canvas = FigureCanvas(fig)
            pl.addWidget(NavToolbar(canvas, page))
            pl.addWidget(canvas)
            # a dotted vertical cursor on the primary axes, moved by the scrubber
            line = fig.axes[0].axvline(self._data["t"][0], color="k", lw=1.0, ls=":")
            self._cursors.append((canvas, line))
            self._tabs.addTab(page, name.capitalize())

    def _setup_scrubber(self, run):
        frames = sorted(glob.glob(os.path.join(run, "frames", "frame_*.jpg")))
        self._frames = []
        self._frame_times = []
        fidx, ftime = self._data["frame_idx"], self._data["t"]
        for p in frames:
            try:
                idx = int(os.path.basename(p).split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            # time of the nearest logged frame index
            j = int(np.argmin(np.abs(fidx - idx)))
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
            self._frame_img.setPixmap(pix.scaledToHeight(236, Qt.SmoothTransformation))
        t = self._frame_times[i]
        self._time_lbl.setText(f"t = {t:.2f} s   (frame {os.path.basename(self._frames[i])})")
        for canvas, line in self._cursors:
            line.set_xdata([t, t])
            canvas.draw_idle()

    def _save(self):
        run = self._selected_run()
        if not run:
            return
        try:
            paths = rr.save_pngs(run)
            self._summary.setText(rr.summary_text(self._data)
                                  + f"\n\nSaved {len(paths)} files → {os.path.join(run, 'viz')}")
        except Exception as exc:
            self._summary.setText(f"Save failed: {exc}")


def main():
    app = QApplication(sys.argv)
    win = RunVisualizer(); win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
