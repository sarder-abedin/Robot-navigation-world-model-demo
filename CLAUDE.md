# CLAUDE.md – Development Guide for AI Assistants

## Project overview

Predictive indoor navigation system for the Freenove FNK0077 Tank Robot.
Split-inference architecture: Pi runs fast AI (YOLOv8n), PC runs heavy AI
(V-JEPA 2 + SSv2 + decision fusion), UI viewer runs on any laptop.

## Repository layout

```
Code/Server/    ← PC AI server (V-JEPA 2, SSv2, decision, TCP server)
Code/Robot/     ← Raspberry Pi client (YOLOv8n, camera, motors, ultrasonic)
Code/Client/    ← Operator UI viewers (display-only, NO AI): Streamlit browser,
                  desktop_viewer.py (native window via pywebview), PyQt5 ai_viewer.py
tests_rpi/      ← Unit tests (no GPU/hardware required)
assets/         ← Demo video clips
```

## Architecture rules

- **PC = TCP server** (binds ports 5003/8003 for UI, 5004/8004 for robot)
- **Pi = TCP client** (connects outbound; **thin client — runs NO AI**)
- **UI viewer = TCP client** (connects to PC only)
- **ALL AI runs on the PC.** YOLOv8n (`Code/Server/detector.py`), V-JEPA 2 and
  SSv2 all run in `Code/Server/` in every mode (live and demo). The Pi streams
  camera frames (port 8004) and the PC detects on them. There is no on-Pi YOLO
  and no `DETECTOR_LOCATION` switch. The Pi's only sensor report is the ultrasonic
  distance (`CMD_SONIC`), its local hard-stop safety.
- The Pi Docker image is therefore lightweight — **no torch / ultralytics**.
- Motor mapping (action → PWM) happens on the **Pi** side in `main_robot.py`,
  not on the PC. The PC sends high-level `CMD_AIMOVE#FORWARD` etc.

## Key files

| File | Purpose |
|---|---|
| `Code/Server/main_server.py` | PC entry point; starts AI pipeline + TCP servers |
| `Code/Server/detector.py` | YOLOv8n — runs on the **PC** on the Pi's streamed frames (all modes) |
| `Code/Server/robot_connection.py` | Parses `CMD_SONIC` from Pi; exposes `get_sonic_cm()` and `send_aimove()` |
| `Code/Server/ai_pipeline.py` | Orchestrates YOLO → V-JEPA 2 → SSv2 → decision → broadcast |
| `Code/Server/robot_control.py` | `TCPRobotController` sends `CMD_AIMOVE` to Pi |
| `Code/Robot/main_robot.py` | Pi entry point (thin client); camera stream + sonic + command loop |
| `Code/Robot/tcp_robot_client.py` | Pi-side TCP client; `send_sonic()` / `send_frame()` |
| `Code/Client/streamlit_viewer.py` | Browser UI (port 8501); bundled in the server Docker image |
| `Code/Client/desktop_viewer.py` | Native desktop window wrapping the Streamlit UI (pywebview); native-only |
| `Code/Client/ai_viewer.py` | PyQt5 UI; AUTO mode (AI drives) / MANUAL mode (operator drives) |

## TCP protocol (summary)

```
Pi → PC:  CMD_SONIC#<sonic_cm>                        (ultrasonic distance; local hard-stop)
Pi → PC:  4-byte LE uint32 + JPEG  (camera stream, port 8004, for ALL server-side AI)
          (the server also still accepts legacy CMD_DETECTION for backward compat)
PC → Pi:  CMD_AIMOVE#<FORWARD|SLOW|STOP|REROUTE>   (AI navigation action)
PC → Pi:  CMD_MOTOR#<L>#<R>                         (manual from UI viewer)
PC → Pi:  CMD_STOP / CMD_KILL / CMD_AIMODE#<0|1|2>
UI → PC:  CMD_AIMODE#<0|1|2>  |  CMD_MOTOR#<L>#<R>  |  CMD_KILL#0
UI → PC:  CMD_LOGGING#<0|1>                         (toggle server-side run logging)
PC → UI:  CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>#<ssv2_sentence>
          (ssv2_sentence = genuine SSv2 label with the YOLO object filled in; last field, optional for old clients)
PC → UI:  4-byte LE uint32 + JPEG  (annotated HUD frames, port 8003)
```

## SSv2 (genuine model) + run logging

- `Code/Server/ssv2_model.py` runs a **real** Something-Something-V2 video
  classifier (VideoMAE, `MCG-NJU/videomae-base-finetuned-ssv2`) over the clip
  buffer. The predicted template's "something" slot is filled with the largest
  obstacle's YOLO class (from the PC's own detection on the streamed frame), e.g.
  "person moving closer". Annotation/logging ONLY — it does NOT drive navigation (the
  fast heuristic in `temporal_action.py` still supplies `temporal_risk`).
  Falls back to a stub (still fills the object) when transformers/weights are absent.
- **Device:** both heavy models (V-JEPA 2 + SSv2) use `device: auto` via
  `device_utils.resolve_device()` → CUDA → MPS → CPU. SSv2's `run_every_n_frames`
  is auto-halved on a GPU. A Docker container on macOS has no Metal passthrough,
  so it runs on CPU (run the server natively for MPS); NVIDIA/DGX uses CUDA with
  `--gpus all`.
- **Run logging is server-side only.** `NavigationLogger` writes CSV + annotated
  frames to the PC's `logs_rpi/`. Initial state: `--logging on|off` flag or
  `NAV_LOGGING=1/0` env; toggled live from the UI (`CMD_LOGGING`).

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
- Import `torch`, `ultralytics`, or any AI framework in `Code/Robot/` — the Pi is
  a thin client with no AI. All models (YOLO/V-JEPA 2/SSv2) live in `Code/Server/`.
- Run blocking calls on the main thread in `AIPipeline` (it runs in a daemon thread).
- Hardcode IP addresses; use config files or CLI flags.
