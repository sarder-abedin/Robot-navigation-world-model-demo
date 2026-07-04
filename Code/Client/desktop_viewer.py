"""
desktop_viewer.py – Run the operator UI in a NATIVE desktop window.

Same Streamlit UI as the browser option (streamlit_viewer.py), but wrapped in an
OS-native window via pywebview so it looks/feels like a desktop app instead of a
browser tab. It:
  1. starts Streamlit headless as a subprocess,
  2. waits for its port to accept connections,
  3. opens a native window pointing at http://127.0.0.1:<port>,
  4. stops Streamlit when the window is closed.

Usage:
  python desktop_viewer.py                 # default port 8501
  python desktop_viewer.py --port 8600 --title "Nav"

Prefer a plain browser tab instead? Just run:
  streamlit run streamlit_viewer.py

Platform notes for the native window (pywebview backend):
  • macOS / Windows: work out of the box (Cocoa / EdgeChromium).
  • Linux: also install a system webview backend, e.g.
      sudo apt-get install -y python3-gi gir1.2-webkit2-4.1   (GTK)
    otherwise use the browser option above.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "streamlit_viewer.py")


def build_streamlit_cmd(port: int, viewer: str = VIEWER) -> list[str]:
    """Command that runs the Streamlit viewer headless on the given port."""
    return [
        sys.executable, "-m", "streamlit", "run", viewer,
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]


def wait_for_port(host: str, port: int, timeout: float = 30.0,
                  interval: float = 0.25) -> bool:
    """Return True once (host, port) accepts a TCP connection, else False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(interval)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Open the Streamlit navigation UI in a native desktop window."
    )
    p.add_argument("--port", type=int, default=8501, help="Streamlit port (default 8501)")
    p.add_argument("--title", default="Predictive Navigation", help="Window title")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=880)
    p.add_argument("--start-timeout", type=float, default=30.0,
                   help="Seconds to wait for Streamlit to come up")
    args = p.parse_args(argv)

    proc = subprocess.Popen(build_streamlit_cmd(args.port))
    try:
        if not wait_for_port("127.0.0.1", args.port, timeout=args.start_timeout):
            print(f"Streamlit did not start on port {args.port} within "
                  f"{args.start_timeout:.0f}s", file=sys.stderr)
            return 1
        # Imported here (not at module top) so --help and the helper functions
        # work on machines without a GUI backend installed.
        import webview  # type: ignore
        print(f"Opening native window → http://127.0.0.1:{args.port}")
        webview.create_window(
            args.title, f"http://127.0.0.1:{args.port}",
            width=args.width, height=args.height,
        )
        webview.start()   # blocks until the window is closed
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
