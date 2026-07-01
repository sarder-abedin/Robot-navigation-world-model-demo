"""
test_robot_connection.py – Unit tests for RobotConnectionServer
(Code/Server/robot_connection.py).

Tests run without a real robot; they use socket pairs to simulate the Pi client.
"""

import sys
import os
import socket
import struct
import threading
import time

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))


@pytest.fixture
def cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "Code", "Server", "config.yaml")
    with open(path) as f:
        c = yaml.safe_load(f)
    # Use high ephemeral ports so tests don't conflict with production
    c["server"]["robot_cmd_port"] = 15004
    c["server"]["robot_video_port"] = 18004
    return c


# ── Mock CameraBuffer ─────────────────────────────────────────────────────────

class MockCameraBuffer:
    def __init__(self):
        self.pushed_frames: list[bytes] = []

    def push_frame(self, jpg: bytes) -> None:
        self.pushed_frames.append(jpg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect_robot(cfg):
    """Simulate the robot connecting on cmd + video sockets."""
    srv = cfg["server"]
    host = "127.0.0.1"
    time.sleep(0.3)  # give the server time to bind

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd_sock.connect((host, srv["robot_cmd_port"]))

    vid_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    vid_sock.connect((host, srv["robot_video_port"]))

    return cmd_sock, vid_sock


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRobotConnectionServer:
    def test_starts_and_accepts_connection(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        server.start()

        cmd_s, vid_s = _connect_robot(cfg)
        time.sleep(0.4)

        assert server.is_connected

        server.stop()
        cmd_s.close()
        vid_s.close()

    def test_sonic_reading_stored(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        server.start()

        cmd_s, vid_s = _connect_robot(cfg)
        time.sleep(0.3)

        cmd_s.sendall(b"CMD_SONIC#88.5\r\n")
        time.sleep(0.15)

        assert abs(server.get_sonic_cm() - 88.5) < 0.01

        server.stop()
        cmd_s.close()
        vid_s.close()

    def test_motor_command_sent_to_robot(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        server.start()

        cmd_s, vid_s = _connect_robot(cfg)
        time.sleep(0.3)
        cmd_s.settimeout(2.0)

        server.send_motor_command(1500, 1500)

        data = b""
        while b"\n" not in data:
            data += cmd_s.recv(128)

        assert b"CMD_MOTOR#1500#1500" in data

        server.stop()
        cmd_s.close()
        vid_s.close()

    def test_video_frame_pushed_to_buffer(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        server.start()

        cmd_s, vid_s = _connect_robot(cfg)
        time.sleep(0.3)

        # Send a synthetic "JPEG" frame
        payload = b"SYNTHETIC_JPEG_DATA"
        vid_s.sendall(struct.pack("<I", len(payload)) + payload)
        time.sleep(0.3)

        # Buffer receives the raw bytes (push_frame decodes it, may return None
        # for non-JPEG but push_frame is still called)
        assert len(buf.pushed_frames) >= 1

        server.stop()
        cmd_s.close()
        vid_s.close()

    def test_send_stop_sends_cmd_stop(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        server.start()

        cmd_s, vid_s = _connect_robot(cfg)
        time.sleep(0.3)
        cmd_s.settimeout(2.0)

        server.send_stop()

        data = b""
        while b"\n" not in data:
            data += cmd_s.recv(128)

        assert b"CMD_STOP" in data

        server.stop()
        cmd_s.close()
        vid_s.close()

    def test_not_connected_returns_false_for_commands(self, cfg):
        from robot_connection import RobotConnectionServer

        buf = MockCameraBuffer()
        server = RobotConnectionServer(cfg, buf)
        # Don't start – no robot
        assert server.is_connected is False
        assert server.send_motor_command(0, 0) is False
        assert server.send_stop() is False
