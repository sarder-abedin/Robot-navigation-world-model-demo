# CLAUDE.md – Development Guide for AI Assistants

## Project overview

Predictive indoor navigation system for the Freenove FNK0077 Tank Robot.
Split-inference architecture: Pi runs fast AI (YOLOv8n), PC runs heavy AI
(V-JEPA 2 + SSv2 + decision fusion), UI viewer runs on any laptop.

## Repository layout

```
Code/Server/    ← PC AI server (V-JEPA 2, SSv2, decision, TCP server)
Code/Robot/     ← Raspberry Pi client (YOLOv8n, camera, motors, ultrasonic)
Code/Client/    ← Operator UI viewer (PyQt5)
tests_rpi/      ← Unit tests (no GPU/hardware required)
assets/         ← Demo video clips
```

## Architecture rules

- **PC = TCP server** (binds ports 5003/8003 for UI, 5004/8004 for robot)
- **Pi = TCP client** (connects outbound; runs YOLOv8n locally)
- **UI viewer = TCP client** (connects to PC only)
- YOLOv8n is NEVER imported in `Code/Server/` during live mode – detection
  comes from the Pi via `CMD_DETECTION`. Local YOLOv8 in `Code/Server/detector.py`
  is ONLY used in demo mode (no robot connected).
- Motor mapping (action → PWM) happens on the **Pi** side in `main_robot.py`,
  not on the PC. The PC sends high-level `CMD_AIMOVE#FORWARD` etc.

## Key files

| File | Purpose |
|---|---|
| `Code/Server/main_server.py` | PC entry point; starts AI pipeline + TCP servers |
| `Code/Server/robot_connection.py` | Parses `CMD_DETECTION` from Pi; exposes `get_latest_detection()` and `send_aimove()` |
| `Code/Server/ai_pipeline.py` | Orchestrates V-JEPA 2 → SSv2 → decision → broadcast |
| `Code/Server/robot_control.py` | `TCPRobotController` sends `CMD_AIMOVE` to Pi |
| `Code/Robot/main_robot.py` | Pi entry point; runs YOLO + camera + sonic + command loop |
| `Code/Robot/detector_robot.py` | YOLOv8n wrapper for Pi; returns `DetectionPacket` |
| `Code/Robot/tcp_robot_client.py` | Pi-side TCP client; `send_detection()` / `send_frame()` |
| `Code/Client/ai_viewer.py` | PyQt5 UI; AUTO mode (AI drives) / MANUAL mode (operator drives) |

## TCP protocol (summary)

```
Pi → PC:  CMD_DETECTION#<risk_pct>#<in_center_0or1>#<area_pct>#<cx_pct>#<sonic_cm>
Pi → PC:  4-byte LE uint32 + JPEG  (camera stream, port 8004, for V-JEPA 2)
PC → Pi:  CMD_AIMOVE#<FORWARD|SLOW|STOP|REROUTE>   (AI navigation action)
PC → Pi:  CMD_MOTOR#<L>#<R>                         (manual from UI viewer)
PC → Pi:  CMD_STOP / CMD_KILL / CMD_AIMODE#<0|1|2>
UI → PC:  CMD_AIMODE#<0|1|2>  |  CMD_MOTOR#<L>#<R>  |  CMD_KILL#0
PC → UI:  CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>
PC → UI:  4-byte LE uint32 + JPEG  (annotated HUD frames, port 8003)
```

## Development workflow

### Run tests (no hardware needed)
```bash
pytest tests_rpi/ -v
```
Pre-existing failures in `test_world_model_rpi.py` are expected when `torch`
is not installed.

### Run demo mode (no robot needed)
```bash
# Place a corridor video at assets/demo_clips/corridor.mp4
cd Code/Server
python main_server.py --mode demo --nav predictive --no-display
```

### Git branch
Always develop on `claude/freenove-predictive-navigation-tgo1v`.

## Config files

- PC: `Code/Server/config.yaml`
- Pi: `Code/Robot/config_robot.yaml`

## Adding new commands

1. Define the wire format string in this file and in `README.md`.
2. Send side: add a `send_xxx()` method in `tcp_robot_client.py` (Pi→PC) or
   `robot_connection.py` (PC→Pi).
3. Receive side: add parsing in `robot_connection.py._cmd_recv_loop()` (Pi→PC)
   or `main_robot.py`'s command loop (PC→Pi).
4. Add tests in `tests_rpi/`.

## Do not

- Import `picamera2`, `gpiozero`, or `lgpio` in server-side code (they are Pi-only).
- Import `ultralytics` in server-side code during live mode (handled by Pi).
- Run blocking calls on the main thread in `AIPipeline` (it runs in a daemon thread).
- Hardcode IP addresses; use config files or CLI flags.
