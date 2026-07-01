"""
test_robot_client.py – Unit tests for RobotTCPClient (Code/Robot/tcp_robot_client.py).

These tests use socket mocks and do NOT require a network connection or hardware.
"""

import sys
import os
import queue
import socket
import struct
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Robot"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_echo_server(host="127.0.0.1", port=0):
    """Bind a listening socket and return (server_sock, bound_port)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    return srv, srv.getsockname()[1]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRobotTCPClientConnect:
    def test_connect_success(self):
        """Client connects successfully when servers are listening."""
        from tcp_robot_client import RobotTCPClient

        cmd_srv, cmd_port = _make_echo_server()
        vid_srv, vid_port = _make_echo_server()

        accepted = []

        def _accept():
            cmd_srv.settimeout(3.0)
            vid_srv.settimeout(3.0)
            accepted.append(cmd_srv.accept()[0])
            accepted.append(vid_srv.accept()[0])

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        client = RobotTCPClient("127.0.0.1", cmd_port, vid_port)
        result = client.connect(timeout=3.0)

        t.join(timeout=3.0)
        assert result is True
        assert client.is_connected

        client.disconnect()
        for s in accepted:
            try:
                s.close()
            except Exception:
                pass
        cmd_srv.close()
        vid_srv.close()

    def test_connect_failure(self):
        """connect() returns False when no server is listening."""
        from tcp_robot_client import RobotTCPClient
        client = RobotTCPClient("127.0.0.1", 59998, 59999)
        result = client.connect(timeout=0.5)
        assert result is False
        assert not client.is_connected

    def test_disconnect_when_not_connected(self):
        """disconnect() on an unconnected client does not raise."""
        from tcp_robot_client import RobotTCPClient
        client = RobotTCPClient("127.0.0.1", 59990, 59991)
        client.disconnect()  # should not raise


class TestRobotTCPClientSendFrame:
    def _setup_connected_client(self):
        """Returns (client, cmd_conn, vid_conn) with an active connection."""
        from tcp_robot_client import RobotTCPClient

        cmd_srv, cmd_port = _make_echo_server()
        vid_srv, vid_port = _make_echo_server()
        conns = []

        def _accept():
            conns.append(cmd_srv.accept()[0])
            conns.append(vid_srv.accept()[0])

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        client = RobotTCPClient("127.0.0.1", cmd_port, vid_port)
        client.connect(timeout=3.0)
        t.join(timeout=3.0)

        cmd_srv.close()
        vid_srv.close()
        return client, conns

    def test_send_frame_sends_length_prefix_plus_data(self):
        """send_frame() prepends a 4-byte LE uint32 length before the JPEG bytes."""
        client, conns = self._setup_connected_client()
        vid_conn = conns[1]
        vid_conn.settimeout(2.0)

        payload = b"FAKEJPEG"
        client.send_frame(payload)

        header = b""
        while len(header) < 4:
            header += vid_conn.recv(4 - len(header))
        length = struct.unpack("<I", header)[0]
        assert length == len(payload)

        body = b""
        while len(body) < length:
            body += vid_conn.recv(length - len(body))
        assert body == payload

        client.disconnect()
        for s in conns:
            try:
                s.close()
            except Exception:
                pass

    def test_send_sonic_format(self):
        """send_sonic() sends CMD_SONIC#<cm> terminated with CRLF."""
        client, conns = self._setup_connected_client()
        cmd_conn = conns[0]
        cmd_conn.settimeout(2.0)

        client.send_sonic(42.7)

        data = b""
        while b"\n" not in data:
            data += cmd_conn.recv(128)

        assert b"CMD_SONIC#42.7" in data

        client.disconnect()
        for s in conns:
            try:
                s.close()
            except Exception:
                pass


class TestRobotTCPClientRecv:
    def test_received_commands_queued(self):
        """Commands sent from the server side appear in get_command()."""
        from tcp_robot_client import RobotTCPClient

        cmd_srv, cmd_port = _make_echo_server()
        vid_srv, vid_port = _make_echo_server()
        cmd_conn_holder = []

        def _accept():
            cmd_conn_holder.append(cmd_srv.accept()[0])
            vid_srv.accept()

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        client = RobotTCPClient("127.0.0.1", cmd_port, vid_port)
        client.connect(timeout=3.0)
        t.join(timeout=3.0)

        cmd_conn = cmd_conn_holder[0]
        cmd_conn.sendall(b"CMD_MOTOR#1500#1500\r\n")
        time.sleep(0.1)

        cmd = client.get_command(timeout=1.0)
        assert cmd is not None
        assert "CMD_MOTOR" in cmd
        assert "1500" in cmd

        client.disconnect()
        cmd_srv.close()
        vid_srv.close()
