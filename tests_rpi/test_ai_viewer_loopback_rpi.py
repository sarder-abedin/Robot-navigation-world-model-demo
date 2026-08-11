"""
test_ai_viewer_loopback_rpi.py – end-to-end loopback through the real recv path.

Pipes real CMD_AISTATUS / CMD_MAPOBJ bytes over a genuine socket into
AIViewer._recv_loop (the actual network thread), so the full client stack runs:
  socket.recv → '\n' framing → pyqtSignal (queued to the GUI thread) →
  _process_status / _process_mapobj → nav_map + world_map accumulation.

This is the piece the offscreen render smoke test didn't cover: the TCP receive
loop and the thread→GUI signal marshalling. No robot, no torch, no display.

Skips when PyQt5 isn't installed (the pure cores are covered by the other tests).
"""

import os
import socket
import time

import pytest

# Qt must render headless (no X display) — set BEFORE QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")

pytest.importorskip("PyQt5", reason="PyQt5 not installed – GUI loopback test skipped")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Client"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Code", "Server"))

from PyQt5 import QtWidgets


def _pump_until(app, cond, timeout=3.0):
    """Spin the Qt event loop (delivering queued signals) until cond() or timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


# Two frames: robot drives 1 m forward; V-JEPA 2 flips to BLOCKED; a wall shows on
# the sonar; a chair is detected (its relative range holds as the robot advances,
# so its WORLD position shifts → flagged moving).
_STATUS_1 = ("CMD_AISTATUS#FORWARD#10#CLEAR#STATIC_CLEAR#-1.0##-1.00#CENTER#none"
             "#-1.00#-1.00#0.0#-1.00#0.000#0.000#0.0")
_MAPOBJ_1 = "CMD_MAPOBJ#chair,0.0,2.00"
_STATUS_2 = ("CMD_AISTATUS#FORWARD#40#BLOCKED#APPROACHING#100.0##1.50#CENTER#none"
             "#-1.00#-1.00#0.0#-1.00#0.000#1.000#0.0")
_MAPOBJ_2 = "CMD_MAPOBJ#chair,0.0,2.00"


def test_loopback_feeds_world_map_over_a_real_socket():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import ai_viewer

    viewer = ai_viewer.AIViewer()

    # A genuine connected socket pair: one end is the viewer's command socket, the
    # other stands in for the PC server pushing status lines.
    server_end, client_end = socket.socketpair()
    try:
        viewer._cmd_sock = client_end
        viewer._connected = True
        import threading
        t = threading.Thread(target=viewer._recv_loop, args=(client_end,),
                             daemon=True, name="TestCmdRecv")
        t.start()

        # Push the two frames. The 2nd CMD_MAPOBJ is split across two writes to
        # prove the recv-loop buffering reassembles a line spanning packets.
        server_end.sendall((_STATUS_1 + "\r\n").encode())
        server_end.sendall((_MAPOBJ_1 + "\r\n").encode())
        server_end.sendall((_STATUS_2 + "\r\n").encode())
        half = len(_MAPOBJ_2) // 2
        server_end.sendall(_MAPOBJ_2[:half].encode())
        time.sleep(0.05)                                    # force a packet boundary
        server_end.sendall((_MAPOBJ_2[half:] + "\r\n").encode())

        m = viewer._world_map._m
        # Wait until BOTH frames are fully applied. The split CMD_MAPOBJ (frame 2)
        # arrives last and is what flips the chair to 'moving' — gating on that
        # avoids a race where the pump exits after frame 1 (chair present, not yet
        # moving) but before the delayed frame-2 line lands.
        ok = _pump_until(app, lambda: len(m.trajectory) >= 2
                         and len(m.foresight_points) >= 1
                         and len(m.objects) >= 1
                         and m.objects[0].moving)
        assert ok, (f"world map didn't fully populate over the socket: "
                    f"traj={len(m.trajectory)} hazards={len(m.foresight_points)} "
                    f"objs={len(m.objects)} "
                    f"moving={[o.moving for o in m.objects]}")

        # Trajectory: the two dead-reckoned poses arrived and anchored the map.
        assert len(m.trajectory) == 2
        assert m.pose is not None
        assert abs(m.pose.y_m - 1.0) < 1e-6

        # Ultrasonic: 100 cm ahead of the robot at (0,1) → world (0, 2).
        assert len(m.obstacle_points) == 1
        ox, oy = m.obstacle_points[0]
        assert abs(ox) < 1e-6 and abs(oy - 2.0) < 1e-6

        # V-JEPA 2 foresight: BLOCKED → hazard at clear_dist 1.5 m ahead of (0,1).
        assert any(h.label == "BLOCKED" for h in m.foresight_points)
        hz = m.foresight_points[-1]
        assert abs(hz.x_m) < 1e-6 and abs(hz.y_m - 2.5) < 1e-6

        # YOLO object survived the SPLIT CMD_MAPOBJ line and was flagged moving
        # (its world position ran from (0,2) to (0,3) as the robot advanced).
        assert len(m.objects) == 1
        assert m.objects[0].label == "chair"
        assert abs(m.objects[0].y_m - 3.0) < 1e-6     # 2 m ahead of pose (0,1)
        assert m.objects[0].moving

        # The status also drove the panel widgets (proves the GUI slot ran).
        assert viewer._action_label.text() == "FORWARD"
        assert viewer._wm_val.text() == "BLOCKED"
    finally:
        viewer._connected = False
        for s in (server_end, client_end):
            try:
                s.close()
            except OSError:
                pass
        app.processEvents()
        viewer.deleteLater()
        app.processEvents()
