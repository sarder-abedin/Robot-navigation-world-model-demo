"""
test_desktop_viewer_rpi.py – unit tests for the native desktop-window launcher.

Covers the non-GUI logic (no window is opened): the Streamlit command it builds
and the port-wait helper. The pywebview import is checked only if installed.
"""

import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Client"))

import desktop_viewer as dv


def test_build_streamlit_cmd_runs_the_viewer_headless():
    cmd = dv.build_streamlit_cmd(8599, viewer="/x/streamlit_viewer.py")
    assert cmd[:5] == [sys.executable, "-m", "streamlit", "run", "/x/streamlit_viewer.py"]
    assert "--server.headless" in cmd and "true" in cmd
    # port is passed through
    assert "--server.port" in cmd and "8599" in cmd
    # bind to loopback (the native window connects to 127.0.0.1)
    i = cmd.index("--server.address")
    assert cmd[i + 1] == "127.0.0.1"


def test_default_viewer_path_points_at_streamlit_viewer():
    assert dv.VIEWER.endswith(os.path.join("Client", "streamlit_viewer.py"))


def test_wait_for_port_true_when_listening():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))          # ephemeral free port
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert dv.wait_for_port("127.0.0.1", port, timeout=2.0) is True
    finally:
        srv.close()


def test_wait_for_port_false_when_closed():
    # grab a port then close it so nothing is listening there
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert dv.wait_for_port("127.0.0.1", port, timeout=0.5, interval=0.1) is False


def test_pywebview_importable_if_installed():
    webview = pytest.importorskip("webview")
    # the two calls desktop_viewer uses must exist
    assert hasattr(webview, "create_window") and hasattr(webview, "start")
