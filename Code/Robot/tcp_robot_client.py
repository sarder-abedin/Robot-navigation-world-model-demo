"""
tcp_robot_client.py – TCP client for the Raspberry Pi robot.

Connects to the PC AI server on two sockets:
  - Command socket (port 5004): bidirectional
      Sends:    CMD_SONIC#<cm>\r\n
      Receives: CMD_MOTOR#<left>#<right>\r\n
                CMD_STOP\r\n
                CMD_KILL\r\n
  - Video socket  (port 8004): robot → PC only
      Sends:    4-byte LE uint32 frame length + JPEG bytes
"""
from __future__ import annotations

import logging
import queue
import socket
import struct
import threading

logger = logging.getLogger(__name__)


class RobotTCPClient:
    """Manages two outbound TCP connections from the robot to the PC server."""

    def __init__(self, server_ip: str, cmd_port: int = 5004, video_port: int = 8004):
        self._server_ip = server_ip
        self._cmd_port = cmd_port
        self._video_port = video_port

        self._cmd_sock: socket.socket | None = None
        self._video_sock: socket.socket | None = None
        self._connected = False
        self._running = False

        self._cmd_lock = threading.Lock()
        self._video_lock = threading.Lock()
        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._recv_thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect both sockets to the PC server. Returns True on success."""
        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.settimeout(timeout)
            self._cmd_sock.connect((self._server_ip, self._cmd_port))
            self._cmd_sock.settimeout(None)
            logger.info("Connected to PC server cmd port %d", self._cmd_port)

            self._video_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._video_sock.settimeout(timeout)
            self._video_sock.connect((self._server_ip, self._video_port))
            self._video_sock.settimeout(None)
            logger.info("Connected to PC server video port %d", self._video_port)

            self._connected = True
            self._running = True
            self._recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True, name="RobotCmdRecv"
            )
            self._recv_thread.start()
            return True

        except Exception as exc:
            logger.error("Connection to PC server failed: %s", exc)
            self.disconnect()
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._running = False
        for s in (self._cmd_sock, self._video_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._cmd_sock = None
        self._video_sock = None

    # ── Send ──────────────────────────────────────────────────────────────────

    def send_frame(self, jpg_bytes: bytes) -> bool:
        """Send a JPEG frame with 4-byte LE length prefix to the PC."""
        if not self._connected or self._video_sock is None:
            return False
        try:
            packet = struct.pack("<I", len(jpg_bytes)) + jpg_bytes
            with self._video_lock:
                self._video_sock.sendall(packet)
            return True
        except Exception as exc:
            logger.error("Frame send failed: %s", exc)
            self._connected = False
            return False

    def send_sonic(self, cm: float) -> bool:
        """Send CMD_SONIC reading to PC."""
        return self._send_cmd(f"CMD_SONIC#{cm:.1f}\r\n")

    # ── Receive ───────────────────────────────────────────────────────────────

    def get_command(self, timeout: float = 0.0) -> str | None:
        """Return next command from PC, or None if queue empty."""
        try:
            return self._cmd_queue.get(block=timeout > 0, timeout=timeout if timeout > 0 else None)
        except queue.Empty:
            return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Private ───────────────────────────────────────────────────────────────

    def _send_cmd(self, msg: str) -> bool:
        if not self._connected or self._cmd_sock is None:
            return False
        try:
            with self._cmd_lock:
                self._cmd_sock.sendall(msg.encode("utf-8"))
            return True
        except Exception as exc:
            logger.error("Cmd send failed: %s", exc)
            self._connected = False
            return False

    def _recv_loop(self) -> None:
        """Receive motor commands from PC on the command socket."""
        buf = ""
        while self._running and self._cmd_sock:
            try:
                data = self._cmd_sock.recv(1024)
                if not data:
                    logger.warning("PC server disconnected")
                    self._connected = False
                    break
                buf += data.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._cmd_queue.put(line)
            except Exception as exc:
                if self._running:
                    logger.error("Robot recv error: %s", exc)
                self._connected = False
                break
