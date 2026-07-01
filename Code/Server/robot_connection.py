"""
robot_connection.py – PC-side TCP server that accepts the robot's connection.

The Raspberry Pi connects outbound to the PC on:
  - port 5004 (robot cmd): bidirectional command channel
      Receives: CMD_SONIC#<cm>\r\n   (ultrasonic readings)
      Sends:    CMD_MOTOR#<L>#<R>\r\n, CMD_STOP\r\n, CMD_AIMODE#<n>\r\n
  - port 8004 (robot video): one-way camera stream
      Receives: 4-byte LE uint32 frame length + JPEG bytes

Once the robot connects this class:
  • Decodes each JPEG frame and pushes it into the shared CameraBuffer
  • Stores the latest ultrasonic reading (polled by AIPipeline)
  • Provides send_motor_command() for the AI decision loop
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

    def send_kill(self) -> bool:
        """Forward CMD_KILL to the robot."""
        return self._send_cmd("CMD_KILL\r\n")

    # ── Sensor API ────────────────────────────────────────────────────────────

    def get_sonic_cm(self) -> float:
        """Return the latest ultrasonic distance received from the robot."""
        with self._sonic_lock:
            return self._latest_sonic_cm

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
        """Accept the robot on both sockets (cmd first, then video)."""
        try:
            self._cmd_server.settimeout(120.0)
            self._video_server.settimeout(120.0)

            logger.info("Accepting robot cmd connection…")
            self._cmd_client, addr = self._cmd_server.accept()
            logger.info("Robot cmd connected from %s", addr)

            logger.info("Accepting robot video connection…")
            self._video_client, addr = self._video_server.accept()
            logger.info("Robot video connected from %s", addr)

            self._connected = True

            self._cmd_recv_thread = threading.Thread(
                target=self._cmd_recv_loop, daemon=True, name="RobotCmdRx"
            )
            self._cmd_recv_thread.start()

            self._video_recv_thread = threading.Thread(
                target=self._video_recv_loop, daemon=True, name="RobotVideoRx"
            )
            self._video_recv_thread.start()

        except Exception as exc:
            if self._running:
                logger.error("Robot accept error: %s", exc)

    def _cmd_recv_loop(self) -> None:
        buf = ""
        while self._running and self._cmd_client:
            try:
                data = self._cmd_client.recv(1024)
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
                    if line.startswith("CMD_SONIC"):
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
        while self._running and self._video_client:
            try:
                header = self._recv_exact(4)
                if header is None:
                    logger.warning("Robot video socket closed")
                    self._connected = False
                    break
                n = struct.unpack("<I", header)[0]
                if n > 10 * 1024 * 1024:
                    logger.error("Oversized robot frame (%d bytes) – dropping", n)
                    self._connected = False
                    break
                jpg = self._recv_exact(n)
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

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n and self._running:
            try:
                chunk = self._video_client.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except Exception:
                return None
        return buf if len(buf) == n else None
