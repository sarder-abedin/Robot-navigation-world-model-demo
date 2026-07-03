"""
streamlit_viewer.py – Browser-based operator UI for the Freenove navigation system.

Open http://localhost:8501 in your browser after starting the Docker container.

Usage (inside Docker, handled by start_server.sh):
    streamlit run Code/Client/streamlit_viewer.py \\
        --server.port 8501 --server.address 0.0.0.0 --server.headless true

Usage (standalone, connecting to a remote server):
    pip install streamlit
    streamlit run Code/Client/streamlit_viewer.py
    # Enter the server IP in the UI
"""
from __future__ import annotations

import socket
import struct
import threading

import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
CMD_PORT   = 5003
VIDEO_PORT = 8003
SPEED_FULL = 1500
SPEED_SLOW = 600

ACTION_BG = {
    "FORWARD": "#1a7a1a",
    "SLOW":    "#c8841a",
    "STOP":    "#8b0000",
    "REROUTE": "#7a3a00",
    "---":     "#444444",
}
WM_COLOR = {
    "BLOCKED": "#ff4444",
    "MIXED":   "#ffaa44",
    "CLEAR":   "#44cc44",
    "UNKNOWN": "#aaaaaa",
}


# ── Backend (one per browser session) ─────────────────────────────────────────

class _Backend:
    """Thread-safe TCP client shared between background recv threads and the UI."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self.running  = False
        self._jpg: bytes | None = None
        self._status  = {
            "action": "---", "risk_pct": 0,
            "wm_label": "UNKNOWN", "pattern": "UNKNOWN", "sonic": "---",
        }
        self._cmd:  socket.socket | None = None
        self._vid:  socket.socket | None = None
        self.log   = "Not connected – enter the server IP and click Connect"
        self.mode  = "AUTO"
        self.vid_frames = 0   # frames received from video socket
        self.cmd_msgs   = 0   # CMD_AISTATUS messages received

    # ── Thread-safe reads (called from the Streamlit main thread) ──────────────

    def get_jpg(self) -> bytes | None:
        with self._lock:
            return self._jpg

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, ip: str) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((ip, CMD_PORT))
            s.settimeout(None)
            self._cmd    = s
            self.running = True
            threading.Thread(target=self._recv_cmd, daemon=True, name="CmdRecv").start()
            # Video connection (non-fatal if unavailable)
            try:
                v = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                v.settimeout(3.0)
                v.connect((ip, VIDEO_PORT))
                v.settimeout(15.0)  # long enough for AI pipeline warmup; catches dead server
                self._vid = v
                threading.Thread(target=self._recv_vid, daemon=True, name="VideoRecv").start()
                self.log = f"Connected to {ip}:{CMD_PORT}  |  video stream active"
            except Exception as e:
                self._vid = None
                self.log = f"Connected to {ip}:{CMD_PORT}  |  video unavailable: {e}"
            return True
        except Exception as e:
            self.log = f"Connection failed: {e}"
            return False

    def disconnect(self) -> None:
        self.running = False
        for s in (self._cmd, self._vid):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._cmd = self._vid = None
        with self._lock:
            self._jpg = None
        self.vid_frames = 0
        self.cmd_msgs = 0
        self.log = "Disconnected"

    # ── Send ───────────────────────────────────────────────────────────────────

    def send(self, msg: str) -> None:
        if not (self._cmd and self.running):
            return
        try:
            self._cmd.sendall(msg.encode("utf-8"))
        except Exception as e:
            self.log = f"Send error: {e}"

    # ── Background recv threads (write to self.* only, no st.* calls) ─────────

    def _recv_cmd(self) -> None:
        buf = ""
        while self.running and self._cmd:
            try:
                raw = self._cmd.recv(1024)
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.startswith("CMD_AISTATUS"):
                        p = line.split("#")
                        if len(p) >= 6:
                            with self._lock:
                                self._status = {
                                    "action":   p[1],
                                    "risk_pct": int(p[2]) if p[2].isdigit() else 0,
                                    "wm_label": p[3],
                                    "pattern":  p[4],
                                    "sonic":    p[5].strip(),
                                }
                            self.cmd_msgs += 1
            except Exception:
                break
        self.running = False

    def _recv_vid(self) -> None:
        def _exact(n: int) -> bytes | None:
            b = b""
            while len(b) < n:
                try:
                    chunk = self._vid.recv(n - len(b))
                except socket.timeout:
                    if not self.running:
                        return None
                    continue  # pipeline may be slow to start; keep waiting
                if not chunk:
                    return None
                b += chunk
            return b

        while self.running and self._vid:
            try:
                hdr = _exact(4)
                if not hdr:
                    break
                length = struct.unpack("<I", hdr)[0]
                jpg = _exact(length)
                if not jpg:
                    break
                with self._lock:
                    self._jpg = jpg
                self.vid_frames += 1
            except Exception:
                break


def _backend() -> _Backend:
    if "backend" not in st.session_state:
        st.session_state.backend = _Backend()
    return st.session_state.backend


# ── Streamlit app ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Freenove Nav – AI Viewer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖  Freenove Predictive Navigation – AI Viewer")

be = _backend()

# ── Connection row ─────────────────────────────────────────────────────────────
c_ip, c_btn, c_log = st.columns([2, 1, 5])

with c_ip:
    ip = st.text_input(
        "PC / Server IP",
        value=st.session_state.get("server_ip", "localhost"),
        key="_ip_input",
        help="Use 'localhost' when the AI server and this viewer run in the same Docker container.",
    )
    st.session_state.server_ip = ip

with c_btn:
    st.write("")
    st.write("")
    if be.running:
        if st.button("Disconnect", use_container_width=True):
            be.disconnect()
            st.rerun()
    else:
        if st.button("Connect", type="primary", use_container_width=True):
            be.connect(ip)
            st.rerun()

with c_log:
    st.write("")
    dot   = "🟢" if be.running else "⚫"
    label = "**CONNECTED**" if be.running else "Disconnected"
    st.markdown(f"{dot} &nbsp; {label} &nbsp; — &nbsp; {be.log}")

st.divider()

# ── Live video + AI state (auto-refreshes every 100 ms) ───────────────────────

@st.fragment(run_every=0.1)
def _live_panel() -> None:
    be_ = _backend()
    jpg = be_.get_jpg()
    s   = be_.get_status()

    v_col, s_col = st.columns([3, 2])

    with v_col:
        st.subheader("Live Video")
        if jpg:
            st.image(jpg, use_container_width=True)
            st.caption(f"Frames received: {be_.vid_frames}")
        else:
            if be_.running:
                connected_ip = st.session_state.get("server_ip", "")
                if connected_ip in ("localhost", "127.0.0.1", "0.0.0.0", ""):
                    st.warning(
                        "Waiting for video… AI pipeline may still be loading. "
                        "If this persists, check that the AI server started "
                        "(look for 'AI pipeline started' in server logs)."
                    )
                else:
                    st.info(
                        f"Waiting for video from {connected_ip}… "
                        "AI pipeline may still be warming up."
                    )
            else:
                st.info("No video – connect to the server to see the camera feed")

    with s_col:
        st.subheader("AI State")

        action = s["action"]
        bg     = ACTION_BG.get(action, "#444")
        st.markdown(
            f"<div style='background:{bg};color:#fff;font-size:26px;font-weight:bold;"
            f"text-align:center;padding:14px;border-radius:6px;margin-bottom:10px'>"
            f"{action}</div>",
            unsafe_allow_html=True,
        )

        r     = s["risk_pct"]
        bar_r = min(r * 2, 255)
        bar_g = min((100 - r) * 2, 255)
        st.markdown(
            f"<div style='margin-bottom:4px'>Risk: <b>{r}%</b></div>"
            f"<div style='background:#333;border-radius:4px;height:18px'>"
            f"<div style='background:rgb({bar_r},{bar_g},0);width:{r}%;"
            f"height:100%;border-radius:4px'></div></div>",
            unsafe_allow_html=True,
        )

        wm   = s["wm_label"]
        wm_c = WM_COLOR.get(wm, "#aaa")
        sonic_raw = s["sonic"]
        try:
            sonic_cm = float(sonic_raw)
            if sonic_cm < 0:
                sonic_str, sonic_c = "---", "#aaa"
            else:
                sonic_str = f"{sonic_cm:.1f} cm"
                sonic_c   = "#ff4444" if sonic_cm < 20 else "#44cc44"
        except ValueError:
            sonic_str = sonic_raw
            sonic_c   = "#aaa"

        st.markdown(
            f"<div style='margin-top:10px;line-height:2.0;font-size:15px'>"
            f"V-JEPA 2: &nbsp;<span style='color:{wm_c};font-weight:bold'>{wm}</span><br>"
            f"Motion: &nbsp;<b>{s['pattern']}</b><br>"
            f"Ultrasonic: &nbsp;<span style='color:{sonic_c};font-weight:bold'>{sonic_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Status messages received: {be_.cmd_msgs}")


_live_panel()

st.divider()

# ── Navigation mode ────────────────────────────────────────────────────────────
st.subheader("Navigation Mode")
nm1, nm2 = st.columns(2)
with nm1:
    if st.button("🔵  PREDICTIVE  (V-JEPA 2 active)", use_container_width=True, type="primary"):
        be.send("CMD_AIMODE#2\n")
        be.log = "Switched to PREDICTIVE mode"
with nm2:
    if st.button("🟡  BASELINE  (reactive, no V-JEPA 2)", use_container_width=True):
        be.send("CMD_AIMODE#1\n")
        be.log = "Switched to BASELINE mode"

st.divider()

# ── Drive control ──────────────────────────────────────────────────────────────
st.subheader("Drive Control")

d1, d2 = st.columns(2)
with d1:
    auto_type = "primary" if be.mode == "AUTO" else "secondary"
    if st.button("🤖  AUTO MODE  (AI drives)", use_container_width=True, type=auto_type):
        be.mode = "AUTO"
        be.send("CMD_AIMODE#2\n")
        be.log = "AUTO MODE – AI decision fuser is driving"
        st.rerun()
with d2:
    man_type = "primary" if be.mode == "MANUAL" else "secondary"
    if st.button("🕹️  MANUAL MODE  (you drive)", use_container_width=True, type=man_type):
        be.mode = "MANUAL"
        be.send("CMD_AIMODE#0\n")
        be.send("CMD_MOTOR#0#0\n")
        be.log = "MANUAL MODE – use the drive buttons below"
        st.rerun()

if be.mode == "MANUAL":
    speed_choice = st.radio(
        "Speed", ["Full  (1500 PWM)", "Slow  (600 PWM)"], horizontal=True
    )
    spd = SPEED_FULL if "Full" in speed_choice else SPEED_SLOW

    _, cfwd, _ = st.columns([1, 1, 1])
    with cfwd:
        if st.button("▲  Forward", use_container_width=True):
            be.send(f"CMD_MOTOR#{spd}#{spd}\n")

    cleft, cstop, cright = st.columns(3)
    with cleft:
        if st.button("◄  Turn Left", use_container_width=True):
            be.send(f"CMD_MOTOR#-{spd}#{spd}\n")
    with cstop:
        if st.button("■  STOP", use_container_width=True):
            be.send("CMD_MOTOR#0#0\n")
    with cright:
        if st.button("Turn Right  ►", use_container_width=True):
            be.send(f"CMD_MOTOR#{spd}#-{spd}\n")

    _, cback, _ = st.columns([1, 1, 1])
    with cback:
        if st.button("▼  Backward", use_container_width=True):
            be.send(f"CMD_MOTOR#-{spd}#-{spd}\n")

    st.caption("Each button sends one motor command. For sustained motion, click repeatedly or hold the button.")

st.divider()

# ── Kill switch ────────────────────────────────────────────────────────────────
st.subheader("🚨  Kill Switch")

ks1, ks2 = st.columns(2)
with ks1:
    if st.button(
        "⚡  EMERGENCY STOP  — halt motors & disable AI",
        type="primary",
        use_container_width=True,
    ):
        be.send("CMD_AIMODE#0\n")
        be.send("CMD_MOTOR#0#0\n")
        be.log = "EMERGENCY STOP sent – motors halted, AI disabled"
        st.toast("EMERGENCY STOP sent!", icon="🛑")
with ks2:
    if st.button(
        "🔴  SHUTDOWN SERVER  — stop the entire server process",
        use_container_width=True,
    ):
        be.send("CMD_KILL#0\n")
        be.log = "Shutdown command sent – server is stopping"
        st.toast("Server shutdown command sent", icon="⚠️")

st.caption(
    "Emergency Stop disables the AI and halts motors but keeps the server running. "
    "Click PREDICTIVE or BASELINE to resume.  |  "
    "Shutdown Server terminates the server process entirely."
)
