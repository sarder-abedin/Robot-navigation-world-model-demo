"""
ai_viewer.py – Operator UI for the Freenove predictive navigation system.

What this viewer does
─────────────────────
1. Connects to the PC navigation server (CMD port 5003, Video port 8003).
2. Shows live AI state: current action, fused risk bar, V-JEPA 2 prediction
   label, motion pattern, and ultrasonic distance.
3. Lets the operator switch between AUTO (AI drives) and MANUAL (operator drives).
4. In MANUAL mode, provides on-screen drive buttons and arrow-key control.
5. Displays the annotated video stream from the server (Video port 8003).
6. Provides an EMERGENCY STOP kill switch – button AND keyboard shortcuts.

Architecture
────────────
  Pi (client)   → runs YOLO11n + camera + motors; sends CMD_DETECTION
  PC (server)   → runs V-JEPA 2 + SSv2 + decision; listens on 5003/8003
  This viewer   → connects to PC on 5003/8003; shows AI state + video

Control modes
─────────────
  AUTO MODE    → PC decision fuser drives the robot via CMD_AIMOVE
  MANUAL MODE  → Operator drives via arrow keys / buttons (CMD_MOTOR relayed by PC)

TCP protocol (send)
───────────────────
  CMD_AIMODE#0          → stop AI, halt motors (STOP AI / MANUAL transition)
  CMD_AIMODE#1          → switch to baseline reactive mode
  CMD_AIMODE#2          → switch to predictive mode
  CMD_MOTOR#<L>#<R>     → manual motor command (−4095 … +4095)
  CMD_KILL#0            → stop AI + robot, shut down server process

TCP protocol (receive)
──────────────────────
  CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>

Kill switch controls
────────────────────
  Space / Escape           → EMERGENCY STOP  (motors stop, AI disabled)
  Ctrl+Q or the button     → SHUTDOWN SERVER (stops the entire demo process)
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
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QRadioButton, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
)

# ── Colour maps ────────────────────────────────────────────────────────────────
ACTION_CSS = {
    "FORWARD":  "background:#1a7a1a; color:white;",
    "SLOW":     "background:#c8841a; color:white;",
    "STOP":     "background:#8b0000; color:white;",
    "REROUTE":  "background:#7a3a00; color:white;",
    "BACKUP":   "background:#a0521a; color:white;",
    "---":      "background:#444;    color:#aaa;",
}
WM_CSS = {
    "BLOCKED": "color:#ff4444;",
    "MIXED":   "color:#ffaa44;",
    "CLEAR":   "color:#44cc44;",
    "UNKNOWN": "color:#aaaaaa;",
}

# Kill-switch button styles
_KILL_READY  = ("background:#cc0000; color:white; font-size:15px; "
                "font-weight:bold; padding:12px; border-radius:4px;")
_KILL_SENT   = ("background:#550000; color:#ff9999; font-size:15px; "
                "font-weight:bold; padding:12px; border-radius:4px;")
_SHUTDOWN_READY = ("background:#4a0000; color:#ffbbbb; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")
_SHUTDOWN_SENT  = ("background:#220000; color:#ff6666; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")

_AUTO_BTN_ACTIVE   = "background:#0a5a8a; color:white; font-weight:bold; padding:6px;"
_AUTO_BTN_INACTIVE = "background:#2a2a2a; color:#aaa; font-weight:bold; padding:6px;"
_MANUAL_BTN_ACTIVE   = "background:#5a3a00; color:white; font-weight:bold; padding:6px;"
_MANUAL_BTN_INACTIVE = "background:#2a2a2a; color:#aaa; font-weight:bold; padding:6px;"

_DRIVE_BTN = ("background:#333; color:white; font-size:16px; "
              "font-weight:bold; padding:10px; border-radius:4px; min-width:60px;")
_DRIVE_STOP_BTN = ("background:#550000; color:white; font-size:14px; "
                   "font-weight:bold; padding:10px; border-radius:4px; min-width:60px;")

CMD_PORT   = 5003
VIDEO_PORT = 8003
MAX_FRAME_BYTES = 10 * 1024 * 1024   # reject a desynced/garbage frame length

SPEED_FULL = 1500
SPEED_SLOW = 600


class AIViewer(QMainWindow):
    # Qt signal so the network recv thread can safely update the UI thread
    status_received = pyqtSignal(str)
    frame_received = pyqtSignal(object)   # QImage from the video thread → main thread
    disconnected = pyqtSignal()           # worker thread requests teardown on the GUI thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Freenove Predictive Navigation – AI Viewer")
        self.resize(720, 800)

        self._cmd_sock: socket.socket | None = None
        self._video_sock: socket.socket | None = None
        self._connected = False
        self._recv_thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None

        self._control_mode: str = "AUTO"   # "AUTO" | "MANUAL"
        self._manual_speed: int = SPEED_FULL
        self._keys_held: set[int] = set()  # avoid repeated motor sends on auto-repeat

        self._build_ui()
        self._register_shortcuts()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_status_bar)
        self._ui_timer.start(200)

        self.status_received.connect(self._process_status)
        self.frame_received.connect(self._show_frame)   # GUI-thread pixmap update
        self.disconnected.connect(self._disconnect)     # GUI-thread teardown

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Connection row ────────────────────────────────────────────────────
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("PC Server IP:"))
        self._ip_edit = QLineEdit("127.0.0.1")
        self._ip_edit.setFixedWidth(145)
        conn_row.addWidget(self._ip_edit)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setFixedWidth(90)
        self._btn_connect.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self._btn_connect)
        conn_row.addStretch()
        root.addLayout(conn_row)

        # ── Video display ─────────────────────────────────────────────────────
        self._video_label = QLabel("[ No video – connect to server ]")
        self._video_label.setFixedSize(420, 315)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet(
            "background:#1a1a1a; border:1px solid #555; color:#666;"
        )
        root.addWidget(self._video_label, alignment=Qt.AlignHCenter)

        # ── AI state panel ────────────────────────────────────────────────────
        state_box = QGroupBox("AI State")
        grid = QtWidgets.QGridLayout(state_box)
        grid.setColumnStretch(1, 1)

        self._action_label = self._make_action_label("---")
        grid.addWidget(QLabel("Action:"),     0, 0)
        grid.addWidget(self._action_label,    0, 1)

        self._risk_bar = QProgressBar()
        self._risk_bar.setRange(0, 100)
        self._risk_bar.setTextVisible(True)
        self._risk_bar.setFormat("Risk: %p%")
        grid.addWidget(QLabel("Risk:"),       1, 0)
        grid.addWidget(self._risk_bar,        1, 1)

        self._wm_val    = self._make_info_val("UNKNOWN")
        self._pat_val   = self._make_info_val("UNKNOWN")
        self._sonic_val = self._make_info_val("---")
        grid.addWidget(QLabel("V-JEPA 2:"),   2, 0)
        grid.addWidget(self._wm_val,           2, 1)
        grid.addWidget(QLabel("Motion:"),      3, 0)
        grid.addWidget(self._pat_val,          3, 1)
        grid.addWidget(QLabel("Sonic:"),       4, 0)
        grid.addWidget(self._sonic_val,        4, 1)

        root.addWidget(state_box)

        # ── Navigation Mode ───────────────────────────────────────────────────
        mode_box = QGroupBox("Navigation Mode  (AI mode only)")
        mode_row = QHBoxLayout(mode_box)

        self._btn_predictive = QPushButton("PREDICTIVE")
        self._btn_predictive.setStyleSheet(
            "background:#1a5f1a; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_predictive.setToolTip("Switch to predictive mode (V-JEPA 2 active)")
        self._btn_predictive.clicked.connect(lambda: self._send_ai_mode(2))
        mode_row.addWidget(self._btn_predictive)

        self._btn_baseline = QPushButton("BASELINE")
        self._btn_baseline.setStyleSheet(
            "background:#7a5500; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_baseline.setToolTip("Switch to baseline reactive mode (no V-JEPA 2)")
        self._btn_baseline.clicked.connect(lambda: self._send_ai_mode(1))
        mode_row.addWidget(self._btn_baseline)

        root.addWidget(mode_box)

        # ── Drive Control ─────────────────────────────────────────────────────
        drive_box = QGroupBox("Drive Control")
        drive_layout = QVBoxLayout(drive_box)
        drive_layout.setSpacing(4)

        # AUTO / MANUAL toggle row
        toggle_row = QHBoxLayout()
        self._btn_auto = QPushButton("AUTO MODE  (AI drives)")
        self._btn_auto.setStyleSheet(_AUTO_BTN_ACTIVE)
        self._btn_auto.setToolTip("Let the AI decision fuser control the robot (Ctrl+A)")
        self._btn_auto.clicked.connect(self._switch_to_auto)
        toggle_row.addWidget(self._btn_auto)

        self._btn_manual = QPushButton("MANUAL MODE  (you drive)")
        self._btn_manual.setStyleSheet(_MANUAL_BTN_INACTIVE)
        self._btn_manual.setToolTip("Take direct control via buttons or arrow keys (Ctrl+M)")
        self._btn_manual.clicked.connect(self._switch_to_manual)
        toggle_row.addWidget(self._btn_manual)
        drive_layout.addLayout(toggle_row)

        # Drive button grid (shown only in MANUAL mode)
        self._drive_widget = QWidget()
        dg = QtWidgets.QGridLayout(self._drive_widget)
        dg.setSpacing(4)

        self._btn_fwd  = self._make_drive_btn("▲", lambda: self._drive_press("FWD"))
        self._btn_back = self._make_drive_btn("▼", lambda: self._drive_press("BACK"))
        self._btn_left = self._make_drive_btn("◄", lambda: self._drive_press("LEFT"))
        self._btn_right = self._make_drive_btn("►", lambda: self._drive_press("RIGHT"))
        self._btn_drive_stop = QPushButton("■ STOP")
        self._btn_drive_stop.setStyleSheet(_DRIVE_STOP_BTN)
        self._btn_drive_stop.clicked.connect(self._drive_stop)

        dg.addWidget(self._btn_fwd,        0, 1)
        dg.addWidget(self._btn_left,       1, 0)
        dg.addWidget(self._btn_drive_stop, 1, 1)
        dg.addWidget(self._btn_right,      1, 2)
        dg.addWidget(self._btn_back,       2, 1)

        # Speed selector
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._radio_full = QRadioButton("Full")
        self._radio_full.setChecked(True)
        self._radio_full.toggled.connect(self._update_speed)
        self._radio_slow = QRadioButton("Slow")
        speed_row.addWidget(self._radio_full)
        speed_row.addWidget(self._radio_slow)
        speed_row.addStretch()
        spd_widget = QWidget()
        spd_widget.setLayout(speed_row)
        dg.addWidget(spd_widget, 3, 0, 1, 3)

        hint = QLabel("Keyboard: Arrow keys drive  |  ↑↓←→ hold to move  |  release to stop")
        hint.setStyleSheet("color:#888; font-size:10px;")
        hint.setAlignment(Qt.AlignCenter)
        dg.addWidget(hint, 4, 0, 1, 3)

        drive_layout.addWidget(self._drive_widget)
        self._drive_widget.setVisible(False)  # hidden in AUTO mode

        root.addWidget(drive_box)

        # ── Kill switch panel ─────────────────────────────────────────────────
        kill_box = QGroupBox("Kill Switch")
        kill_box.setStyleSheet(
            "QGroupBox { border:2px solid #aa0000; border-radius:4px; "
            "margin-top:6px; font-weight:bold; color:#ff6666; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }"
        )
        kill_layout = QVBoxLayout(kill_box)
        kill_layout.setSpacing(4)

        self._btn_kill = QPushButton("EMERGENCY STOP   [Space / Esc]")
        self._btn_kill.setStyleSheet(_KILL_READY)
        self._btn_kill.setToolTip(
            "Immediately stops all motors and disables AI navigation.\n"
            "Server stays running; resume with PREDICTIVE or BASELINE.\n"
            "Keyboard: Space or Escape"
        )
        self._btn_kill.clicked.connect(self._emergency_stop)
        kill_layout.addWidget(self._btn_kill)

        self._btn_shutdown = QPushButton("SHUTDOWN SERVER   [Ctrl+Q]")
        self._btn_shutdown.setStyleSheet(_SHUTDOWN_READY)
        self._btn_shutdown.setToolTip(
            "Stops motors AND shuts down the server process.\n"
            "Use this to end the demo completely.\n"
            "Keyboard: Ctrl+Q"
        )
        self._btn_shutdown.clicked.connect(self._shutdown_server)
        kill_layout.addWidget(self._btn_shutdown)

        hint2 = QLabel(
            "Space / Esc = emergency stop motors   |   Ctrl+Q = shutdown server"
        )
        hint2.setStyleSheet("color:#884444; font-size:10px;")
        hint2.setAlignment(Qt.AlignCenter)
        kill_layout.addWidget(hint2)

        root.addWidget(kill_box)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QLabel("Not connected – enter the PC server IP and click Connect")
        self._status_bar.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self._status_bar)

    def _make_action_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Monospace", 13, QFont.Bold))
        lbl.setStyleSheet(ACTION_CSS.get(text, ACTION_CSS["---"]))
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumWidth(180)
        return lbl

    def _make_info_val(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Monospace", 11))
        lbl.setStyleSheet("color:#aaa;")
        return lbl

    def _make_drive_btn(self, symbol: str, slot) -> QPushButton:
        btn = QPushButton(symbol)
        btn.setStyleSheet(_DRIVE_BTN)
        btn.pressed.connect(slot)
        btn.released.connect(self._drive_stop)
        return btn

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────

    def _register_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space),  self, self._emergency_stop)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._emergency_stop)
        QShortcut(QKeySequence("Ctrl+Q"), self, self._shutdown_server)
        QShortcut(QKeySequence("Ctrl+P"), self, lambda: self._send_ai_mode(2))
        QShortcut(QKeySequence("Ctrl+B"), self, lambda: self._send_ai_mode(1))
        QShortcut(QKeySequence("Ctrl+A"), self, self._switch_to_auto)
        QShortcut(QKeySequence("Ctrl+M"), self, self._switch_to_manual)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._control_mode != "MANUAL" or event.isAutoRepeat():
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            if key not in self._keys_held:
                self._keys_held.add(key)
                self._apply_key_drive()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._control_mode != "MANUAL" or event.isAutoRepeat():
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            self._keys_held.discard(key)
            if not self._keys_held:
                self._send_motor(0, 0)
            else:
                self._apply_key_drive()
        else:
            super().keyReleaseEvent(event)

    def _apply_key_drive(self) -> None:
        """Send motor command based on the set of currently-held arrow keys."""
        s = self._manual_speed
        if Qt.Key_Up in self._keys_held and Qt.Key_Left in self._keys_held:
            self._send_motor(s // 2, s)
        elif Qt.Key_Up in self._keys_held and Qt.Key_Right in self._keys_held:
            self._send_motor(s, s // 2)
        elif Qt.Key_Down in self._keys_held and Qt.Key_Left in self._keys_held:
            self._send_motor(-s, -s // 2)
        elif Qt.Key_Down in self._keys_held and Qt.Key_Right in self._keys_held:
            self._send_motor(-s // 2, -s)
        elif Qt.Key_Up in self._keys_held:
            self._send_motor(s, s)
        elif Qt.Key_Down in self._keys_held:
            self._send_motor(-s, -s)
        elif Qt.Key_Left in self._keys_held:
            self._send_motor(-s, s)
        elif Qt.Key_Right in self._keys_held:
            self._send_motor(s, -s)

    # ── Drive button actions ───────────────────────────────────────────────────

    def _drive_press(self, direction: str) -> None:
        s = self._manual_speed
        mapping = {
            "FWD":   (s, s),
            "BACK":  (-s, -s),
            "LEFT":  (-s, s),
            "RIGHT": (s, -s),
        }
        L, R = mapping.get(direction, (0, 0))
        self._send_motor(L, R)

    def _drive_stop(self) -> None:
        if not self._keys_held:  # don't interrupt keyboard driving
            self._send_motor(0, 0)

    def _update_speed(self) -> None:
        self._manual_speed = SPEED_FULL if self._radio_full.isChecked() else SPEED_SLOW

    # ── Drive control mode ─────────────────────────────────────────────────────

    def _switch_to_auto(self) -> None:
        self._control_mode = "AUTO"
        self._keys_held.clear()
        self._drive_widget.setVisible(False)
        self._btn_auto.setStyleSheet(_AUTO_BTN_ACTIVE)
        self._btn_manual.setStyleSheet(_MANUAL_BTN_INACTIVE)
        # Resume predictive AI
        self._send_ai_mode(2)
        self._status_bar.setText("AUTO MODE – AI decision fuser is driving")

    def _switch_to_manual(self) -> None:
        self._control_mode = "MANUAL"
        self._drive_widget.setVisible(True)
        self._btn_auto.setStyleSheet(_AUTO_BTN_INACTIVE)
        self._btn_manual.setStyleSheet(_MANUAL_BTN_ACTIVE)
        # Disable AI motors; PC relays manual CMD_MOTOR from now on
        self._send_ai_mode(0)
        self._send_motor(0, 0)
        self._status_bar.setText(
            "MANUAL MODE – arrow keys / buttons drive  |  Ctrl+A = return to AUTO"
        )

    # ── Kill switch actions ────────────────────────────────────────────────────

    def _emergency_stop(self) -> None:
        self._send_ai_mode(0)
        self._send_motor(0, 0)
        self._btn_kill.setText("STOPPED  ✓  – click PREDICTIVE / BASELINE to resume")
        self._btn_kill.setStyleSheet(_KILL_SENT)
        self._status_bar.setText(
            "EMERGENCY STOP sent – motors halted, AI disabled. "
            "Click PREDICTIVE or BASELINE to resume."
        )
        QTimer.singleShot(4000, self._reset_kill_button)

    def _shutdown_server(self) -> None:
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot send shutdown command.")
            return
        try:
            self._cmd_sock.sendall(b"CMD_KILL#0\n")
            self._btn_shutdown.setText("SHUTDOWN SENT  ✓")
            self._btn_shutdown.setStyleSheet(_SHUTDOWN_SENT)
            self._status_bar.setText(
                "SHUTDOWN command sent – server is stopping. "
                "Connection will drop in a moment."
            )
        except Exception as exc:
            self._status_bar.setText(f"Shutdown send error: {exc}")

    def _reset_kill_button(self) -> None:
        self._btn_kill.setText("EMERGENCY STOP   [Space / Esc]")
        self._btn_kill.setStyleSheet(_KILL_READY)

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
            self._status_bar.setText(
                f"Connected to {ip}:{CMD_PORT}  |  "
                "Space/Esc = stop   Ctrl+Q = shutdown   Ctrl+M = manual"
            )
            self._recv_thread = threading.Thread(
                target=self._recv_loop, args=(self._cmd_sock,), daemon=True, name="CmdRecv"
            )
            self._recv_thread.start()

            # Video connection (non-fatal if unavailable)
            self._video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._video_sock.settimeout(3.0)
                self._video_sock.connect((ip, VIDEO_PORT))
                self._video_sock.settimeout(1.0)
                self._video_thread = threading.Thread(
                    target=self._video_loop, args=(self._video_sock,), daemon=True, name="VideoRecv"
                )
                self._video_thread.start()
            except Exception:
                try:
                    self._video_sock.close()
                except Exception:
                    pass
                self._video_sock = None
                self._video_label.setText("[ Video unavailable ]")

        except Exception as exc:
            try:
                if self._cmd_sock:
                    self._cmd_sock.close()
            except Exception:
                pass
            self._cmd_sock = None
            self._status_bar.setText(f"Connection failed: {exc}")

    def _disconnect(self) -> None:
        self._connected = False
        self._keys_held.clear()
        for s in (self._cmd_sock, self._video_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._cmd_sock = self._video_sock = None
        self._btn_connect.setText("Connect")
        self._video_label.setText("[ No video – connect to server ]")
        self._status_bar.setText("Disconnected")

    # ── Network threads ────────────────────────────────────────────────────────

    def _recv_loop(self, sock) -> None:
        # Bind to the socket this thread was started with; if a reconnect swaps
        # self._cmd_sock, this stale thread stops instead of clobbering the new one.
        buf = ""
        while self._connected and sock is self._cmd_sock:
            try:
                raw = sock.recv(1024)
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("CMD_AISTATUS"):
                        self.status_received.emit(line)
            except Exception:
                break
        if sock is self._cmd_sock:            # only tear down the CURRENT connection
            self.disconnected.emit()          # marshalled to the GUI thread

    def _video_loop(self, sock) -> None:
        while self._connected and sock is self._video_sock:
            header = self._recv_exact(sock, 4)
            if header is None:
                break
            n = struct.unpack("<I", header)[0]
            if not (0 < n <= MAX_FRAME_BYTES):   # desynced/garbage length → give up
                break
            jpg = self._recv_exact(sock, n)
            if jpg is None:
                break
            try:
                arr = np.frombuffer(jpg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                # .copy() so the QImage owns its buffer (rgb goes out of scope) and
                # is safe to hand to the GUI thread, which builds the QPixmap.
                qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
                self.frame_received.emit(qimg)
            except Exception:
                continue

    def _show_frame(self, qimg) -> None:
        """GUI-thread slot: turn the QImage into a scaled pixmap."""
        pix = QPixmap.fromImage(qimg).scaled(
            420, 315, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._video_label.setPixmap(pix)

    def _recv_exact(self, sock, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                # The pipeline can be slow to emit the first frame; keep waiting as
                # long as we're still the active connection instead of dying.
                if self._connected and sock is self._video_sock:
                    continue
                return None
            except Exception:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    # ── Status display ─────────────────────────────────────────────────────────

    def _process_status(self, line: str) -> None:
        # CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>
        parts = line.split("#")
        if len(parts) < 6:
            return
        _, action, risk_pct, wm_label, pattern, sonic = parts[:6]

        self._action_label.setText(action)
        self._action_label.setStyleSheet(ACTION_CSS.get(action, ACTION_CSS["---"]))

        try:
            pct = int(risk_pct)
            self._risk_bar.setValue(pct)
            r = min(pct * 2, 255)
            g = min((100 - pct) * 2, 255)
            self._risk_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: rgb({r},{g},0); }}"
            )
        except ValueError:
            pass

        self._wm_val.setText(wm_label)
        self._wm_val.setStyleSheet(WM_CSS.get(wm_label, WM_CSS["UNKNOWN"]))

        self._pat_val.setText(pattern)

        sonic = sonic.strip()
        try:
            cm = float(sonic)
            if cm < 0:
                # -1 = no echo / sensor blind — don't render it as a 0 cm obstacle.
                self._sonic_val.setText("--- (no echo)")
                self._sonic_val.setStyleSheet("color:#888;")
            else:
                self._sonic_val.setText(f"{cm:.1f} cm")
                self._sonic_val.setStyleSheet(
                    "color:#ff4444;" if cm < 20 else "color:#44cc44;"
                )
        except ValueError:
            self._sonic_val.setText(sonic)

    def _update_status_bar(self) -> None:
        if self._connected:
            mode_hint = "AUTO" if self._control_mode == "AUTO" else "MANUAL (↑↓←→)"
            self._status_bar.setText(
                f"Connected  [{mode_hint}]  |  "
                "Space/Esc = emergency stop   Ctrl+Q = shutdown   Ctrl+M/A = mode"
            )

    # ── Command helpers ────────────────────────────────────────────────────────

    def _send_ai_mode(self, mode: int) -> None:
        """Send CMD_AIMODE#<mode> to the server."""
        if not (self._cmd_sock and self._connected):
            self._status_bar.setText("Not connected – cannot send command.")
            return
        try:
            self._cmd_sock.sendall(f"CMD_AIMODE#{mode}\n".encode("utf-8"))
        except Exception as exc:
            self._status_bar.setText(f"Send error: {exc}")

    def _send_motor(self, left: int, right: int) -> None:
        """Send CMD_MOTOR#<L>#<R> to the server (relayed to robot in manual mode)."""
        if not (self._cmd_sock and self._connected):
            return
        try:
            self._cmd_sock.sendall(f"CMD_MOTOR#{left}#{right}\n".encode("utf-8"))
        except Exception as exc:
            self._status_bar.setText(f"Motor cmd error: {exc}")

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._disconnect()
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window,          QColor(45,  45,  45))
    palette.setColor(QtGui.QPalette.WindowText,      QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base,            QColor(30,  30,  30))
    palette.setColor(QtGui.QPalette.AlternateBase,   QColor(50,  50,  50))
    palette.setColor(QtGui.QPalette.Text,            QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button,          QColor(55,  55,  55))
    palette.setColor(QtGui.QPalette.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Highlight,       QColor(0,   120, 215))
    palette.setColor(QtGui.QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = AIViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
