"""
ai_viewer.py – Lightweight AI status viewer for the Freenove tank robot.

This is a minimal PyQt5 window that:
  1. Connects to the predictive navigation server (CMD port 5003).
  2. Displays live AI state: action, risk, V-JEPA 2 label, motion pattern,
     ultrasonic distance.
  3. Lets the operator switch navigation modes (Predictive / Baseline / Stop).
  4. Optionally displays the annotated video stream (Video port 8003).

The heavy AI workload stays entirely on the server (Raspberry Pi).
This client is intentionally lightweight – no ML model is loaded here.

Protocol:
  SEND:   CMD_AIMODE#0          → stop AI, manual control
          CMD_AIMODE#1          → switch to baseline reactive mode
          CMD_AIMODE#2          → switch to predictive mode
  RECV:   CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>
"""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# ── Status colours ─────────────────────────────────────────────────────────────
ACTION_CSS = {
    "FORWARD":  "background:#1a7a1a; color:white;",
    "SLOW":     "background:#c8841a; color:white;",
    "STOP":     "background:#8b0000; color:white;",
    "REROUTE":  "background:#7a3a00; color:white;",
    "---":      "background:#444; color:#aaa;",
}
WM_CSS = {
    "BLOCKED": "color:#ff4444;",
    "MIXED":   "color:#ffaa44;",
    "CLEAR":   "color:#44cc44;",
    "UNKNOWN": "color:#aaaaaa;",
}

CMD_PORT = 5003
VIDEO_PORT = 8003


class AIViewer(QMainWindow):
    status_received = pyqtSignal(str)  # emitted from network thread to UI thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Freenove Predictive Navigation – AI Viewer")
        self.resize(680, 520)

        self._cmd_sock: socket.socket | None = None
        self._video_sock: socket.socket | None = None
        self._connected = False
        self._recv_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None

        self._build_ui()

        # Poll for status messages at 10 Hz
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_status_bar)
        self._ui_timer.start(100)

        self.status_received.connect(self._process_status)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── Connection bar ────────────────────────────────────────────────────
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Robot IP:"))
        self._ip_edit = QLineEdit("192.168.0.100")
        self._ip_edit.setFixedWidth(140)
        conn_row.addWidget(self._ip_edit)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self._btn_connect)
        conn_row.addStretch()
        root.addLayout(conn_row)

        # ── Video display ─────────────────────────────────────────────────────
        self._video_label = QLabel()
        self._video_label.setFixedSize(400, 300)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet("background:#222; border:1px solid #555;")
        self._video_label.setText("[ No video ]")
        root.addWidget(self._video_label, alignment=Qt.AlignHCenter)

        # ── AI state panel ────────────────────────────────────────────────────
        state_box = QGroupBox("AI State")
        state_grid = QtWidgets.QGridLayout(state_box)

        self._action_label = self._big_label("---")
        state_grid.addWidget(QLabel("Action:"), 0, 0)
        state_grid.addWidget(self._action_label, 0, 1)

        self._risk_bar = QProgressBar()
        self._risk_bar.setRange(0, 100)
        self._risk_bar.setTextVisible(True)
        self._risk_bar.setFormat("Risk: %p%")
        state_grid.addWidget(QLabel("Risk:"), 1, 0)
        state_grid.addWidget(self._risk_bar, 1, 1)

        self._wm_label = self._info_label("V-JEPA2:", "UNKNOWN")
        state_grid.addWidget(self._wm_label[0], 2, 0)
        state_grid.addWidget(self._wm_label[1], 2, 1)

        self._pattern_lbl = self._info_label("Motion:", "UNKNOWN")
        state_grid.addWidget(self._pattern_lbl[0], 3, 0)
        state_grid.addWidget(self._pattern_lbl[1], 3, 1)

        self._sonic_lbl = self._info_label("Sonic:", "---")
        state_grid.addWidget(self._sonic_lbl[0], 4, 0)
        state_grid.addWidget(self._sonic_lbl[1], 4, 1)

        root.addWidget(state_box)

        # ── Mode control buttons ──────────────────────────────────────────────
        mode_row = QHBoxLayout()
        self._btn_predictive = QPushButton("PREDICTIVE MODE")
        self._btn_predictive.setStyleSheet(
            "background:#1a5f1a; color:white; font-weight:bold;"
        )
        self._btn_predictive.clicked.connect(lambda: self._send_ai_mode(2))
        mode_row.addWidget(self._btn_predictive)

        self._btn_baseline = QPushButton("BASELINE MODE")
        self._btn_baseline.setStyleSheet(
            "background:#7a5500; color:white; font-weight:bold;"
        )
        self._btn_baseline.clicked.connect(lambda: self._send_ai_mode(1))
        mode_row.addWidget(self._btn_baseline)

        self._btn_stop_ai = QPushButton("STOP AI / MANUAL")
        self._btn_stop_ai.setStyleSheet(
            "background:#7a0000; color:white; font-weight:bold;"
        )
        self._btn_stop_ai.clicked.connect(lambda: self._send_ai_mode(0))
        mode_row.addWidget(self._btn_stop_ai)
        root.addLayout(mode_row)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar_label = QLabel("Not connected")
        self._status_bar_label.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(self._status_bar_label)

    def _big_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Monospace", 14, QFont.Bold))
        lbl.setStyleSheet(ACTION_CSS.get(text, ACTION_CSS["---"]))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedWidth(200)
        return lbl

    def _info_label(self, caption: str, text: str):
        cap = QLabel(caption)
        val = QLabel(text)
        val.setFont(QFont("Monospace", 11))
        val.setStyleSheet("color:#aaa;")
        return cap, val

    # ── Connection ─────────────────────────────────────────────────────────────

    def _toggle_connection(self) -> None:
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        ip = self._ip_edit.text().strip()
        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.settimeout(3.0)
            self._cmd_sock.connect((ip, CMD_PORT))
            self._cmd_sock.settimeout(None)
            self._connected = True
            self._btn_connect.setText("Disconnect")
            self._status_bar_label.setText(f"Connected to {ip}:{CMD_PORT}")

            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True
            )
            self._recv_thread.start()

            # Also connect video
            self._video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._video_sock.settimeout(3.0)
                self._video_sock.connect((ip, VIDEO_PORT))
                self._video_sock.settimeout(1.0)
                self._video_thread = threading.Thread(
                    target=self._video_loop, daemon=True
                )
                self._video_thread.start()
            except Exception:
                self._video_sock = None

        except Exception as exc:
            self._status_bar_label.setText(f"Connection failed: {exc}")

    def _disconnect(self) -> None:
        self._connected = False
        for s in (self._cmd_sock, self._video_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._cmd_sock = self._video_sock = None
        self._btn_connect.setText("Connect")
        self._status_bar_label.setText("Disconnected")

    # ── Network threads ────────────────────────────────────────────────────────

    def _recv_loop(self) -> None:
        buf = ""
        while self._connected and self._cmd_sock:
            try:
                data = self._cmd_sock.recv(1024).decode("utf-8", errors="replace")
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("CMD_AISTATUS"):
                        self.status_received.emit(line)
            except Exception:
                break
        self._disconnect()

    def _video_loop(self) -> None:
        while self._connected and self._video_sock:
            try:
                raw_len = self._recv_exact(4)
                if not raw_len:
                    break
                n = struct.unpack("<I", raw_len)[0]
                jpg = self._recv_exact(n)
                if not jpg:
                    break
                arr = np.frombuffer(jpg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qimg).scaled(
                        400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self._video_label.setPixmap(pix)
            except Exception:
                break

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = self._video_sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except Exception:
                return None
        return buf

    # ── Status processing ──────────────────────────────────────────────────────

    def _process_status(self, line: str) -> None:
        # CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>
        parts = line.split("#")
        if len(parts) < 6:
            return
        _, action, risk_pct, wm_label, pattern, sonic = parts[:6]

        # Action label
        self._action_label.setText(action)
        self._action_label.setStyleSheet(ACTION_CSS.get(action, ACTION_CSS["---"]))

        # Risk bar
        try:
            self._risk_bar.setValue(int(risk_pct))
            r = int(risk_pct) * 2
            g = (100 - int(risk_pct)) * 2
            self._risk_bar.setStyleSheet(
                f"QProgressBar::chunk {{background: rgb({min(r,255)},{min(g,255)},0);}}"
            )
        except ValueError:
            pass

        # World model label
        self._wm_label[1].setText(wm_label)
        self._wm_label[1].setStyleSheet(WM_CSS.get(wm_label, WM_CSS["UNKNOWN"]))

        # Motion pattern
        self._pattern_lbl[1].setText(pattern)

        # Ultrasonic
        sonic = sonic.strip()
        try:
            cm = float(sonic)
            self._sonic_lbl[1].setText(f"{cm:.1f} cm")
            self._sonic_lbl[1].setStyleSheet(
                "color:#ff4444;" if cm < 20 else "color:#44cc44;"
            )
        except ValueError:
            self._sonic_lbl[1].setText(sonic)

    def _update_status_bar(self) -> None:
        if not self._connected:
            return
        self._status_bar_label.setText(
            f"Connected – updates via CMD_AISTATUS"
        )

    # ── Mode commands ──────────────────────────────────────────────────────────

    def _send_ai_mode(self, mode: int) -> None:
        if self._cmd_sock and self._connected:
            try:
                self._cmd_sock.sendall(f"CMD_AIMODE#{mode}\n".encode("utf-8"))
            except Exception as exc:
                self._status_bar_label.setText(f"Send error: {exc}")

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = AIViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
