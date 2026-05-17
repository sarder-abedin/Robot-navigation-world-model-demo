"""
ai_viewer.py – AI-capable client for the Freenove predictive navigation demo.

Architecture (post-refactor)
────────────────────────────
Heavy AI runs HERE on the operator PC / laptop:
  • V-JEPA 2 world model  (transformer, benefits from GPU)
  • SSv2-style temporal pattern recogniser
  • Decision fusion (weighted risk → FORWARD / SLOW / STOP / REROUTE)

The Raspberry Pi handles only:
  • Camera capture + YOLOv8 obstacle detection
  • Ultrasonic safety guard
  • Motor execution (receives CMD_AIMOVE from this client)

Threading model
───────────────
  CmdRecv   – reads TCP command socket; parses CMD_DETECTION messages
  VideoRecv – reads length-prefixed JPEG frames; feeds rolling frame buffer
  AIInfer   – runs WorldModel + Temporal + Decision; sends CMD_AIMOVE to Pi
  Qt main   – UI event loop

TCP protocol
────────────
  SEND:
    CMD_AIMODE#0/1/2   → stop AI / baseline / predictive (Pi acknowledges)
    CMD_AIMOVE#<ACT>   → navigation action from client AI (Pi executes)
    CMD_KILL#0         → stop motors + shut down server process
  RECV:
    CMD_DETECTION#<yolo_risk_pct>#<obs_in_center>#<area_frac_pct>
                      #<centroid_x_pct>#<sonic_cm>

Kill switch controls
────────────────────
  Space / Escape     → EMERGENCY STOP  (CMD_AIMODE#0, motors halt, AI paused)
  Ctrl+Q / button    → SHUTDOWN SERVER (CMD_KILL#0, server process exits)
  Ctrl+P             → switch to PREDICTIVE mode
  Ctrl+B             → switch to BASELINE mode
"""

from __future__ import annotations

import logging
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QShortcut, QVBoxLayout, QWidget,
)

# ── Client-side AI modules (same dir as this file) ────────────────────────────
_CLIENT_DIR = Path(__file__).parent
sys.path.insert(0, str(_CLIENT_DIR))

from decision import DecisionFuser, Action, DecisionResult          # noqa: E402
from temporal_action import TemporalActionRecognizer, FrameObstacleState  # noqa: E402
from world_model import WorldModel, WorldModelResult                # noqa: E402

logger = logging.getLogger(__name__)

# ── Style constants ───────────────────────────────────────────────────────────
ACTION_CSS = {
    "FORWARD":  "background:#1a7a1a; color:white;",
    "SLOW":     "background:#c8841a; color:white;",
    "STOP":     "background:#8b0000; color:white;",
    "REROUTE":  "background:#7a3a00; color:white;",
    "---":      "background:#444;    color:#aaa;",
}
WM_CSS = {
    "BLOCKED": "color:#ff4444;",
    "MIXED":   "color:#ffaa44;",
    "CLEAR":   "color:#44cc44;",
    "UNKNOWN": "color:#aaaaaa;",
}
_KILL_READY     = ("background:#cc0000; color:white; font-size:15px; "
                   "font-weight:bold; padding:12px; border-radius:4px;")
_KILL_SENT      = ("background:#550000; color:#ff9999; font-size:15px; "
                   "font-weight:bold; padding:12px; border-radius:4px;")
_SHUTDOWN_READY = ("background:#4a0000; color:#ffbbbb; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")
_SHUTDOWN_SENT  = ("background:#220000; color:#ff6666; font-size:11px; "
                   "font-weight:bold; padding:6px; border-radius:3px;")

CMD_PORT   = 5003
VIDEO_PORT = 8003

# ── Config ────────────────────────────────────────────────────────────────────
_FALLBACK_CFG: dict = {
    "navigation_mode": "predictive",
    "camera": {"clip_length": 16},
    "world_model": {
        "model_id": "facebook/vjepa2-vitl-fpc64-256",
        "input_size": 224,
        "prediction_horizon": 4,
        "risk_similarity_threshold": 0.55,
        "run_every_n_frames": 8,
        "device": "cpu",
    },
    "temporal_action": {"window_size": 10, "approach_ratio": 0.6, "clear_ratio": 0.7},
    "decision": {
        "weights": {"detector": 0.35, "world_model": 0.45, "temporal": 0.20},
        "low_risk_max": 0.30,
        "medium_risk_max": 0.60,
        "hysteresis": 0.05,
        "stop_hold_seconds": 1.5,
    },
    "robot": {"ultrasonic_stop_cm": 15.0},
}


def _load_cfg() -> dict:
    cfg_path = _CLIENT_DIR / "config_client.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            loaded = yaml.safe_load(f) or {}
        # Merge with fallback so any missing keys still work
        merged = dict(_FALLBACK_CFG)
        for k, v in loaded.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return _FALLBACK_CFG


# ── Detection state (parsed from CMD_DETECTION) ────────────────────────────────
class _DetectionMsg:
    """One parsed CMD_DETECTION message from the Pi."""
    __slots__ = ("yolo_risk", "obs_in_center", "area_frac", "centroid_x", "sonic_cm")

    def __init__(
        self,
        yolo_risk: float = 0.0,
        obs_in_center: bool = False,
        area_frac: float = 0.0,
        centroid_x: float = 0.5,
        sonic_cm: float = -1.0,
    ):
        self.yolo_risk = yolo_risk
        self.obs_in_center = obs_in_center
        self.area_frac = area_frac
        self.centroid_x = centroid_x
        self.sonic_cm = sonic_cm

    def to_temporal_state(self) -> FrameObstacleState:
        return FrameObstacleState(
            obstacle_present=self.area_frac > 0.005,
            in_center=self.obs_in_center,
            area_frac=self.area_frac,
            centroid_x=self.centroid_x,
        )


class AIViewer(QMainWindow):
    # Signals for cross-thread UI updates
    ai_updated = pyqtSignal(str, int, str, str, float, float, float)
    # args: action, risk_pct, wm_label, pattern, sonic_cm, wm_risk, temporal_risk

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Freenove Predictive Navigation – AI Client")
        self.resize(740, 760)

        self._cfg = _load_cfg()
        self._clip_len = self._cfg["camera"]["clip_length"]
        self._sonic_stop = self._cfg["robot"]["ultrasonic_stop_cm"]
        self._nav_mode = self._cfg.get("navigation_mode", "predictive")

        # Network state
        self._cmd_sock: socket.socket | None = None
        self._video_sock: socket.socket | None = None
        self._connected = False

        # AI state
        self._ai_active = True
        self._ai_running = False

        # Rolling frame buffer for V-JEPA 2 (filled by video thread)
        self._frame_lock = threading.Lock()
        self._frame_buf: deque[np.ndarray] = deque(maxlen=self._clip_len)

        # Latest detection from Pi (filled by cmd recv thread)
        self._det_lock = threading.Lock()
        self._latest_det: _DetectionMsg | None = None

        # Client-side AI components (initialised in _start_ai())
        self._world_model: WorldModel | None = None
        self._temporal: TemporalActionRecognizer | None = None
        self._fuser: DecisionFuser | None = None

        self._build_ui()
        self._register_shortcuts()

        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_status_bar)
        self._ui_timer.start(200)

        self.ai_updated.connect(self._on_ai_updated)

        # Start AI models loading in background so the UI is responsive
        t = threading.Thread(target=self._start_ai, daemon=True, name="AIInit")
        t.start()

    # ── AI initialisation ──────────────────────────────────────────────────────

    def _start_ai(self) -> None:
        """Load AI models in a background thread (can take several seconds)."""
        logger.info("Loading client-side AI models…")
        cfg = self._cfg
        self._world_model = WorldModel(cfg)
        self._world_model.load()
        self._temporal = TemporalActionRecognizer(cfg)
        self._fuser = DecisionFuser(cfg, self._nav_mode)
        self._ai_running = True
        logger.info("Client AI ready (mode=%s)", self._nav_mode)

    # ── AI inference thread ────────────────────────────────────────────────────

    def _ai_loop(self) -> None:
        """
        Runs at ~10 Hz.  Pulls the latest frame clip + detection, runs the full
        predictive AI stack, then sends CMD_AIMOVE to the Pi.
        """
        while self._connected:
            if not self._ai_running:
                time.sleep(0.1)
                continue

            # Snapshot frame buffer
            with self._frame_lock:
                clip = list(self._frame_buf)

            # Snapshot latest detection
            with self._det_lock:
                det = self._latest_det

            if det is None:
                time.sleep(0.05)
                continue

            # V-JEPA 2 world model
            if clip:
                wm_result = self._world_model.predict(clip)
            else:
                wm_result = WorldModelResult(
                    buffer_ready=False, predicted_risk=det.yolo_risk
                )
            wm_risk = wm_result.predicted_risk if wm_result.buffer_ready else det.yolo_risk

            # SSv2 temporal recogniser
            self._temporal.push(det.to_temporal_state())
            temporal_result = self._temporal.classify()

            # Ultrasonic risk (sonic_cm < stop threshold → hard risk 1.0)
            sonic_risk = (
                1.0
                if (det.sonic_cm > 0 and det.sonic_cm < self._sonic_stop)
                else 0.0
            )

            # Decision fusion
            decision = self._fuser.decide(
                detector_risk=det.yolo_risk,
                world_model_risk=wm_risk,
                temporal_risk=temporal_result.temporal_risk,
                world_model_label=wm_result.label,
                temporal_pattern=temporal_result.pattern,
                ultrasonic_risk=sonic_risk,
            )

            logger.debug(
                "AI: %s risk=%.2f | det=%.2f wm=%.2f ta=%.2f",
                decision.action, decision.risk_score,
                det.yolo_risk, wm_risk, temporal_result.temporal_risk,
            )

            # Send navigation action to Pi
            if self._ai_active:
                self._send_aimove(decision.action)

            # Update UI
            self.ai_updated.emit(
                str(decision.action),
                int(decision.risk_score * 100),
                wm_result.label,
                str(temporal_result.pattern),
                det.sonic_cm,
                wm_risk,
                temporal_result.temporal_risk,
            )

            time.sleep(0.10)  # ~10 Hz AI loop

    # ── UI builder ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Connection row ────────────────────────────────────────────────────
        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Robot IP:"))
        self._ip_edit = QLineEdit("192.168.0.100")
        self._ip_edit.setFixedWidth(145)
        conn_row.addWidget(self._ip_edit)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.setFixedWidth(90)
        self._btn_connect.clicked.connect(self._toggle_connection)
        conn_row.addWidget(self._btn_connect)
        self._ai_status_lbl = QLabel("AI: loading models…")
        self._ai_status_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        conn_row.addWidget(self._ai_status_lbl)
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
        state_box = QGroupBox("AI State  (V-JEPA 2 + SSv2 + Decision — running on this PC)")
        grid = QtWidgets.QGridLayout(state_box)
        grid.setColumnStretch(1, 1)

        self._action_label = self._make_action_label("---")
        grid.addWidget(QLabel("Action:"),      0, 0)
        grid.addWidget(self._action_label,     0, 1)

        self._risk_bar = QProgressBar()
        self._risk_bar.setRange(0, 100)
        self._risk_bar.setTextVisible(True)
        self._risk_bar.setFormat("Fused Risk: %p%")
        grid.addWidget(QLabel("Fused Risk:"), 1, 0)
        grid.addWidget(self._risk_bar,         1, 1)

        self._yolo_bar = QProgressBar()
        self._yolo_bar.setRange(0, 100)
        self._yolo_bar.setFormat("YOLOv8 (Pi): %p%")
        self._yolo_bar.setStyleSheet(
            "QProgressBar::chunk { background: rgb(0,140,255); }"
        )
        grid.addWidget(QLabel("YOLO (Pi):"),  2, 0)
        grid.addWidget(self._yolo_bar,         2, 1)

        self._wm_bar = QProgressBar()
        self._wm_bar.setRange(0, 100)
        self._wm_bar.setFormat("V-JEPA 2 (PC): %p%")
        self._wm_bar.setStyleSheet(
            "QProgressBar::chunk { background: rgb(180,80,200); }"
        )
        grid.addWidget(QLabel("V-JEPA 2:"),   3, 0)
        grid.addWidget(self._wm_bar,           3, 1)

        self._ta_bar = QProgressBar()
        self._ta_bar.setRange(0, 100)
        self._ta_bar.setFormat("Temporal (PC): %p%")
        self._ta_bar.setStyleSheet(
            "QProgressBar::chunk { background: rgb(200,140,0); }"
        )
        grid.addWidget(QLabel("Temporal:"),    4, 0)
        grid.addWidget(self._ta_bar,           4, 1)

        self._wm_val    = self._make_info_val("UNKNOWN")
        self._pat_val   = self._make_info_val("UNKNOWN")
        self._sonic_val = self._make_info_val("---")
        grid.addWidget(QLabel("V-JEPA 2:"),   5, 0)
        grid.addWidget(self._wm_val,           5, 1)
        grid.addWidget(QLabel("Motion:"),      6, 0)
        grid.addWidget(self._pat_val,          6, 1)
        grid.addWidget(QLabel("Sonic:"),       7, 0)
        grid.addWidget(self._sonic_val,        7, 1)

        root.addWidget(state_box)

        # ── Mode control ──────────────────────────────────────────────────────
        mode_box = QGroupBox("Navigation Mode")
        mode_row = QHBoxLayout(mode_box)

        self._btn_predictive = QPushButton("PREDICTIVE  [Ctrl+P]")
        self._btn_predictive.setStyleSheet(
            "background:#1a5f1a; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_predictive.setToolTip(
            "Enable V-JEPA 2 world model + full SSv2 weights (predictive mode)"
        )
        self._btn_predictive.clicked.connect(lambda: self._send_ai_mode(2))
        mode_row.addWidget(self._btn_predictive)

        self._btn_baseline = QPushButton("BASELINE  [Ctrl+B]")
        self._btn_baseline.setStyleSheet(
            "background:#7a5500; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_baseline.setToolTip(
            "Disable V-JEPA 2; react to YOLOv8 + temporal only (baseline mode)"
        )
        self._btn_baseline.clicked.connect(lambda: self._send_ai_mode(1))
        mode_row.addWidget(self._btn_baseline)

        self._btn_stop_ai = QPushButton("STOP AI / MANUAL")
        self._btn_stop_ai.setStyleSheet(
            "background:#7a0000; color:white; font-weight:bold; padding:6px;"
        )
        self._btn_stop_ai.setToolTip(
            "Pause AI; motors stop. Manual CMD_MOTOR commands pass through."
        )
        self._btn_stop_ai.clicked.connect(lambda: self._send_ai_mode(0))
        mode_row.addWidget(self._btn_stop_ai)

        root.addWidget(mode_box)

        # ── Kill switch ───────────────────────────────────────────────────────
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
            "Immediately stops all motors and pauses AI.\n"
            "Server stays alive; click PREDICTIVE or BASELINE to resume.\n"
            "Keyboard: Space or Escape"
        )
        self._btn_kill.clicked.connect(self._emergency_stop)
        kill_layout.addWidget(self._btn_kill)

        self._btn_shutdown = QPushButton("SHUTDOWN SERVER   [Ctrl+Q]")
        self._btn_shutdown.setStyleSheet(_SHUTDOWN_READY)
        self._btn_shutdown.setToolTip(
            "Stops motors AND shuts down the server process on the Pi.\n"
            "Use this to end the demo completely.\n"
            "Keyboard: Ctrl+Q"
        )
        self._btn_shutdown.clicked.connect(self._shutdown_server)
        kill_layout.addWidget(self._btn_shutdown)

        hint = QLabel(
            "Space / Esc = emergency stop   |   Ctrl+Q = shutdown server"
        )
        hint.setStyleSheet("color:#884444; font-size:10px;")
        hint.setAlignment(Qt.AlignCenter)
        kill_layout.addWidget(hint)

        root.addWidget(kill_box)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QLabel(
            "Not connected – enter the Raspberry Pi IP and click Connect"
        )
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

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────

    def _register_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space),  self, self._emergency_stop)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._emergency_stop)
        QShortcut(QKeySequence("Ctrl+Q"),      self, self._shutdown_server)
        QShortcut(QKeySequence("Ctrl+P"),      self, lambda: self._send_ai_mode(2))
        QShortcut(QKeySequence("Ctrl+B"),      self, lambda: self._send_ai_mode(1))

    # ── Kill switch actions ────────────────────────────────────────────────────

    def _emergency_stop(self) -> None:
        self._send_ai_mode(0)
        self._btn_kill.setText("STOPPED  ✓  – click PREDICTIVE / BASELINE to resume")
        self._btn_kill.setStyleSheet(_KILL_SENT)
        self._status_bar.setText(
            "EMERGENCY STOP sent – robot motors halted, AI paused. "
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
                "Space/Esc = stop   Ctrl+Q = shutdown"
            )

            # Command receive thread (CMD_DETECTION messages from Pi)
            threading.Thread(
                target=self._recv_loop, daemon=True, name="CmdRecv"
            ).start()

            # AI inference thread (runs V-JEPA2 + temporal + decision)
            threading.Thread(
                target=self._ai_loop, daemon=True, name="AIInfer"
            ).start()

            # Video receive thread
            self._video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self._video_sock.settimeout(3.0)
                self._video_sock.connect((ip, VIDEO_PORT))
                self._video_sock.settimeout(1.0)
                threading.Thread(
                    target=self._video_loop, daemon=True, name="VideoRecv"
                ).start()
            except Exception:
                self._video_sock = None
                self._video_label.setText("[ Video unavailable ]")

        except Exception as exc:
            self._cmd_sock = None
            self._status_bar.setText(f"Connection failed: {exc}")

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
        self._video_label.setText("[ No video – connect to server ]")
        self._status_bar.setText("Disconnected")

    # ── Network receive threads ────────────────────────────────────────────────

    def _recv_loop(self) -> None:
        """Read CMD_DETECTION and other messages from the server command socket."""
        buf = ""
        while self._connected and self._cmd_sock:
            try:
                raw = self._cmd_sock.recv(2048)
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("CMD_DETECTION"):
                        self._parse_detection(line)
            except Exception:
                break
        self._disconnect()

    def _parse_detection(self, line: str) -> None:
        """
        Parse CMD_DETECTION#<yolo_risk_pct>#<obs_in_center>#<area_frac_pct>
                            #<centroid_x_pct>#<sonic_cm>
        and store as latest detection state for the AI inference thread.
        """
        try:
            parts = line.split("#")
            if len(parts) < 6:
                return
            yolo_risk    = int(parts[1]) / 100.0
            obs_in_center = bool(int(parts[2]))
            area_frac    = int(parts[3]) / 100.0
            centroid_x   = int(parts[4]) / 100.0
            sonic_cm     = float(parts[5])

            det = _DetectionMsg(
                yolo_risk=yolo_risk,
                obs_in_center=obs_in_center,
                area_frac=area_frac,
                centroid_x=centroid_x,
                sonic_cm=sonic_cm,
            )
            # Also update YOLO bar directly from network thread via signal
            with self._det_lock:
                self._latest_det = det
            # Update yolo risk bar (safe because Qt signal not needed for int/float)
            # We emit through ai_updated at the AI loop rate instead.
        except Exception as exc:
            logger.debug("CMD_DETECTION parse error: %s – line: %s", exc, line)

    def _video_loop(self) -> None:
        """Receive length-prefixed JPEG frames and push to rolling frame buffer."""
        while self._connected and self._video_sock:
            try:
                header = self._recv_exact(4)
                if not header:
                    break
                n = struct.unpack("<I", header)[0]
                jpg = self._recv_exact(n)
                if not jpg:
                    break
                arr = np.frombuffer(jpg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue

                # Resize to AI frame size and add to rolling buffer
                h_target = self._cfg["world_model"]["input_size"]
                frame_rgb = cv2.cvtColor(
                    cv2.resize(bgr, (h_target, h_target)),
                    cv2.COLOR_BGR2RGB,
                )
                with self._frame_lock:
                    self._frame_buf.append(frame_rgb)

                # Display full-size frame in UI
                rgb_disp = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_disp.shape
                qimg = QImage(rgb_disp.data, w, h, w * ch, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    420, 315, Qt.KeepAspectRatio, Qt.SmoothTransformation
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

    # ── UI update (called from Qt main thread via signal) ─────────────────────

    def _on_ai_updated(
        self,
        action: str,
        risk_pct: int,
        wm_label: str,
        pattern: str,
        sonic_cm: float,
        wm_risk: float,
        temporal_risk: float,
    ) -> None:
        # Action
        self._action_label.setText(action)
        self._action_label.setStyleSheet(ACTION_CSS.get(action, ACTION_CSS["---"]))

        # Fused risk bar
        self._risk_bar.setValue(risk_pct)
        r = min(risk_pct * 2, 255)
        g = min((100 - risk_pct) * 2, 255)
        self._risk_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: rgb({r},{g},0); }}"
        )

        # Component risk bars
        with self._det_lock:
            det = self._latest_det
        if det:
            self._yolo_bar.setValue(int(det.yolo_risk * 100))
        self._wm_bar.setValue(int(wm_risk * 100))
        self._ta_bar.setValue(int(temporal_risk * 100))

        # Labels
        self._wm_val.setText(wm_label)
        self._wm_val.setStyleSheet(WM_CSS.get(wm_label, WM_CSS["UNKNOWN"]))
        self._pat_val.setText(pattern)

        # Sonic
        if sonic_cm > 0:
            self._sonic_val.setText(f"{sonic_cm:.1f} cm")
            self._sonic_val.setStyleSheet(
                "color:#ff4444;" if sonic_cm < 20 else "color:#44cc44;"
            )
        else:
            self._sonic_val.setText("---")

        # AI status label
        if self._ai_running:
            self._ai_status_lbl.setText(f"AI: {self._nav_mode.upper()}")
            self._ai_status_lbl.setStyleSheet(
                "color:#44cc44; font-size:10px;" if self._ai_active
                else "color:#888; font-size:10px;"
            )

    def _update_status_bar(self) -> None:
        if not self._connected:
            return
        buf_size = len(self._frame_buf)
        clip_len = self._clip_len
        if not self._ai_running:
            self._status_bar.setText(
                "Connected – waiting for AI models to finish loading…"
            )
        else:
            self._status_bar.setText(
                f"Connected  |  Buf: {buf_size}/{clip_len}  "
                f"|  Mode: {self._nav_mode.upper()}  "
                f"|  Space/Esc = stop   Ctrl+Q = shutdown"
            )

    # ── Mode and action commands ───────────────────────────────────────────────

    def _send_ai_mode(self, mode: int) -> None:
        """
        Switch navigation mode.

        mode 0 → stop AI; mode 1 → baseline; mode 2 → predictive.

        The mode change affects the client's DecisionFuser weights.
        The server is also notified so it can log the mode.
        """
        if mode == 0:
            self._ai_active = False
        elif mode == 1:
            self._ai_active = True
            self._nav_mode = "baseline"
            if self._ai_running:
                self._fuser = DecisionFuser(self._cfg, "baseline")
        elif mode == 2:
            self._ai_active = True
            self._nav_mode = "predictive"
            if self._ai_running:
                self._fuser = DecisionFuser(self._cfg, "predictive")

        if self._cmd_sock and self._connected:
            try:
                self._cmd_sock.sendall(f"CMD_AIMODE#{mode}\n".encode("utf-8"))
            except Exception as exc:
                self._status_bar.setText(f"Send error: {exc}")
        else:
            self._status_bar.setText("Not connected – cannot send command.")

    def _send_aimove(self, action: Action) -> None:
        """Send CMD_AIMOVE#<ACTION> to the Pi to execute the motor command."""
        if not (self._cmd_sock and self._connected):
            return
        try:
            self._cmd_sock.sendall(f"CMD_AIMOVE#{action.value}\n".encode("utf-8"))
        except Exception as exc:
            logger.debug("CMD_AIMOVE send error: %s", exc)

    # ── Window close ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._disconnect()
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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
