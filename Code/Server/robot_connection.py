"""
robot_connection.py – PC-side TCP server that accepts the robot's connection.

The Raspberry Pi connects outbound to the PC on:
  - port 5004 (robot cmd): bidirectional command channel
      Receives: CMD_DETECTION#<risk_pct>#<in_center>#<area_pct>#<cx_pct>#<sonic_cm>
                CMD_SONIC#<cm>  (legacy; still accepted)
      Sends:    CMD_AIMOVE#<FORWARD|SLOW|STOP|REROUTE>
                CMD_MOTOR#<L>#<R>  (manual mode)
                CMD_STOP, CMD_AIMODE#<n>, CMD_KILL
  - port 8004 (robot video): one-way camera stream
      Receives: 4-byte LE uint32 frame length + JPEG bytes

Once the robot connects this class:
  • Decodes each JPEG frame and pushes it into the shared CameraBuffer
  • Stores the latest CMD_DETECTION result (polled by AIPipeline)
  • Provides get_latest_detection() returning a DetectionResult for the pipeline
  • Provides send_aimove() for the AI decision loop
  • Provides send_motor_command() for manual UI control
"""
from __future__ import annotations

import logging
import queue
import socket
import struct
import threading

logger = logging.getLogger(__name__)


class RobotConnectionServer:
    """PC-side TCP server that waits for the Pi robot to connect."""

    def __init__(self, cfg: dict, camera_buffer=None):
        srv = cfg.get("server", {})
        self._robot_cmd_port = srv.get("robot_cmd_port", 5004)
        self._robot_video_port = srv.get("robot_video_port", 8004)
        self._camera_buffer = camera_buffer

        self._cmd_server: socket.socket | None = None
        self._video_server: socket.socket | None = None
        self._cmd_client: socket.socket | None = None
        self._video_client: socket.socket | None = None

        self._connected = False
        self._running = False
        self._cmd_lock = threading.Lock()

        self._latest_sonic_cm: float = -1.0
        self._sonic_lock = threading.Lock()

        # Detection result received from Pi via CMD_DETECTION
        self._latest_detection: dict = {
            "yolo_risk_pct": 0,
            "obs_in_center": False,
            "area_frac_pct": 0,
            "centroid_x_pct": 50,
            "n_obstacles": 0,
            "top_label": "",
        }
        self._detection_lock = threading.Lock()

        self._extra_cmd_queue: queue.Queue[str] = queue.Queue()

        self._accept_thread: threading.Thread | None = None
        self._cmd_recv_thread: threading.Thread | None = None
        self._video_recv_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, host: str = "0.0.0.0") -> None:
        """Bind listening sockets and start the accept thread."""
        self._running = True

        self._cmd_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cmd_server.bind((host, self._robot_cmd_port))
        self._cmd_server.listen(1)

        self._video_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._video_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._video_server.bind((host, self._robot_video_port))
        self._video_server.listen(1)

        logger.info("Waiting for robot on cmd=%d  video=%d",
                    self._robot_cmd_port, self._robot_video_port)

        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="RobotAccept"
        )
        self._accept_thread.start()

    def stop(self) -> None:
        self._running = False
        self._connected = False
        for s in (self._cmd_client, self._video_client,
                  self._cmd_server, self._video_server):
            try:
                if s:
                    s.close()
            except Exception:
                pass

    # ── Motor command API (called by AI pipeline) ─────────────────────────────

    def send_motor_command(self, left: int, right: int) -> bool:
        """Send CMD_MOTOR#<left>#<right> to the robot."""
        return self._send_cmd(f"CMD_MOTOR#{int(left)}#{int(right)}\r\n")

    def send_stop(self) -> bool:
        """Send CMD_STOP to the robot (hard safety halt)."""
        return self._send_cmd("CMD_STOP\r\n")

    def send_aimode(self, mode: int) -> bool:
        """Forward a CMD_AIMODE change to the robot."""
        return self._send_cmd(f"CMD_AIMODE#{mode}\r\n")

    def send_aimove(self, action: str) -> bool:
        """
        Send an AI navigation action to the robot.

        The Pi maps the action string to motor PWM calls locally:
          FORWARD  → full-speed forward
          SLOW     → slow forward
          STOP     → halt
          REROUTE  → back-up + spin maneuver
        """
        return self._send_cmd(f"CMD_AIMOVE#{action}\r\n")

    def send_kill(self) -> bool:
        """Forward CMD_KILL to the robot."""
        return self._send_cmd("CMD_KILL\r\n")

    # ── Sensor / Detection API ────────────────────────────────────────────────

    def get_sonic_cm(self) -> float:
        """Return the latest ultrasonic distance received from the robot."""
        with self._sonic_lock:
            return self._latest_sonic_cm

    def get_latest_detection(self):
        """
        Return the most recent CMD_DETECTION result as a DetectionResult
        compatible with the server-side AI pipeline.

        Reconstructs a synthetic bounding box from the aggregated area/centroid
        fields so that the temporal action recogniser has a box to work with.
        """
        import math
        from detector import DetectionResult

        with self._detection_lock:
            d = dict(self._latest_detection)

        risk = d["yolo_risk_pct"] / 100.0
        area_frac = d["area_frac_pct"] / 100.0
        cx_norm = d["centroid_x_pct"] / 100.0
        obs_in_center = d["obs_in_center"]
        n = d["n_obstacles"]
        top_label = d.get("top_label", "") or "obstacle"

        boxes = []
        if n > 0 and area_frac > 0:
            frame_area = 400 * 300
            side = math.sqrt(area_frac * frame_area)
            cx = int(cx_norm * 400)
            cy = 150
            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(400, int(cx + side / 2))
            y2 = min(300, int(cy + side / 2))
            if x2 > x1 and y2 > y1:
                boxes = [(x1, y1, x2, y2)]

        return DetectionResult(
            boxes=boxes,
            labels=[top_label] * len(boxes),
            confidences=[risk] * len(boxes),
            obstacle_in_center=obs_in_center,
            closest_area=area_frac,
            raw_risk=risk,
            frame_width=400,
            frame_height=300,
        )

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Private ───────────────────────────────────────────────────────────────

    def _send_cmd(self, msg: str) -> bool:
        if not self._connected or self._cmd_client is None:
            return False
        try:
            with self._cmd_lock:
                self._cmd_client.sendall(msg.encode("utf-8"))
            return True
        except Exception as exc:
            logger.error("Motor cmd send failed: %s", exc)
            self._connected = False
            return False

    def _accept_loop(self) -> None:
        """Accept the robot on both sockets; re-accept after disconnect."""
        self._cmd_server.settimeout(5.0)
        self._video_server.settimeout(5.0)

        while self._running:
            try:
                logger.info("Waiting for robot cmd connection…")
                try:
                    self._cmd_client, addr = self._cmd_server.accept()
                except socket.timeout:
                    continue
                logger.info("Robot cmd connected from %s", addr)

                logger.info("Waiting for robot video connection…")
                try:
                    self._video_client, addr = self._video_server.accept()
                except socket.timeout:
                    try:
                        self._cmd_client.close()
                    except Exception:
                        pass
                    self._cmd_client = None
                    continue
                logger.info("Robot video connected from %s", addr)

                self._connected = True

                rx_cmd = threading.Thread(
                    target=self._cmd_recv_loop, daemon=True, name="RobotCmdRx"
                )
                rx_vid = threading.Thread(
                    target=self._video_recv_loop, daemon=True, name="RobotVideoRx"
                )
                self._cmd_recv_thread = rx_cmd
                self._video_recv_thread = rx_vid
                rx_cmd.start()
                rx_vid.start()

                # Wait until EITHER recv thread exits. If only one channel drops
                # (e.g. the command socket errors while video keeps streaming),
                # tearing down both prevents a half-connected state where the UI
                # shows live video but send_aimove() silently fails because
                # _connected is False and the robot never moves.
                while self._running and rx_cmd.is_alive() and rx_vid.is_alive():
                    rx_cmd.join(timeout=0.5)

                self._connected = False
                # Force the still-alive channel to unblock by closing its socket,
                # then wait for both threads to finish before re-accepting.
                for s in (self._cmd_client, self._video_client):
                    try:
                        if s:
                            s.close()
                    except Exception:
                        pass
                rx_cmd.join(timeout=2.0)
                rx_vid.join(timeout=2.0)

                logger.info("Robot disconnected; ready to re-accept")
                self._cmd_client = None
                self._video_client = None

            except Exception as exc:
                if self._running:
                    logger.error("Robot accept loop error: %s", exc)

    def _cmd_recv_loop(self) -> None:
        # Bind the socket once at thread start. If this thread outlives its
        # connection and _accept_loop re-accepts, self._cmd_client points at the
        # NEW socket — reading self._cmd_client here would steal the new
        # connection's data. Using the captured local avoids that.
        sock = self._cmd_client
        buf = ""
        while self._running and sock is self._cmd_client:
            try:
                data = sock.recv(1024)
                if not data:
                    logger.warning("Robot cmd socket closed")
                    self._connected = False
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("CMD_DETECTION"):
                        # CMD_DETECTION#<risk_pct>#<in_center>#<area_pct>#<cx_pct>#<sonic_cm>
                        parts = line.split("#")
                        if len(parts) >= 6:
                            try:
                                risk_pct = int(parts[1])
                                in_center = parts[2].strip() == "1"
                                area_pct = int(parts[3])
                                cx_pct = int(parts[4])
                                sonic_cm = float(parts[5])
                                # Optional trailing YOLO label (SSv2 filler).
                                top_label = parts[6].strip() if len(parts) >= 7 else ""
                                with self._detection_lock:
                                    self._latest_detection = {
                                        "yolo_risk_pct": risk_pct,
                                        "obs_in_center": in_center,
                                        "area_frac_pct": area_pct,
                                        "centroid_x_pct": cx_pct,
                                        "n_obstacles": 1 if risk_pct > 0 else 0,
                                        "top_label": top_label,
                                    }
                                with self._sonic_lock:
                                    self._latest_sonic_cm = sonic_cm
                            except (ValueError, IndexError):
                                pass
                    elif line.startswith("CMD_SONIC"):
                        # Legacy sonic-only message
                        parts = line.split("#")
                        if len(parts) >= 2:
                            try:
                                cm = float(parts[1])
                                with self._sonic_lock:
                                    self._latest_sonic_cm = cm
                            except ValueError:
                                pass
                    else:
                        self._extra_cmd_queue.put(line)
            except Exception as exc:
                if self._running:
                    logger.error("Robot cmd recv error: %s", exc)
                self._connected = False
                break

    def _video_recv_loop(self) -> None:
        # Capture the socket once (see _cmd_recv_loop) so a stale thread can't
        # read the next connection's video stream after a re-accept.
        sock = self._video_client
        while self._running and sock is self._video_client:
            try:
                header = self._recv_exact(sock, 4)
                if header is None:
                    logger.warning("Robot video socket closed")
                    self._connected = False
                    break
                n = struct.unpack("<I", header)[0]
                if n > 10 * 1024 * 1024:
                    logger.error("Oversized robot frame (%d bytes) – dropping", n)
                    self._connected = False
                    break
                jpg = self._recv_exact(sock, n)
                if jpg is None:
                    self._connected = False
                    break
                if self._camera_buffer is not None:
                    self._camera_buffer.push_frame(jpg)
            except Exception as exc:
                if self._running:
                    logger.error("Robot video recv error: %s", exc)
                self._connected = False
                break

    def _recv_exact(self, sock, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n and self._running:
            try:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except Exception:
                return None
        return buf if len(buf) == n else None
