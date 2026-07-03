# Freenove Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot Kit for
Raspberry Pi (FNK0077)** that uses **V-JEPA 2** as a world model to anticipate
future obstacles — not just react to what is currently visible.

---

## Quick Start

### Fastest path — Docker on Mac / Linux (no robot needed)

```bash
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# Build the server image (arm64 Mac Apple Silicon + amd64 Linux both work)
docker build -f Dockerfile.server -t nav-server .

# Live mode — server waits for Pi to connect on ports 5004/8004
docker run --rm \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 \
  nav-server python main_server.py --mode live --nav predictive --no-display

# Demo mode — no robot needed; supply any corridor video first:
#   mkdir -p assets/demo_clips && cp /path/to/corridor.mp4 assets/demo_clips/
docker run --rm \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 \
  -v "$(pwd)/assets:/app/assets:ro" \
  -v "$(pwd)/logs_rpi:/app/logs_rpi" \
  nav-server
```

> **V-JEPA 2 weights** (~300 MB) are downloaded from HuggingFace automatically
> on first run.  No GPU required — CPU-only inference works out of the box.

### Fastest path — Raspberry Pi robot (Docker)

```bash
# On the Pi: install camera/GPIO host libs once
sudo apt-get install -y python3-picamera2 python3-libcamera python3-kms++

# Build the robot image (cross-compile on Mac/Linux or build directly on Pi)
docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .

# Run (replace 192.168.1.42 with your PC's IP)
docker run --rm --privileged \
  --device /dev/video0 \
  --device /dev/gpiochip4 \
  -e SERVER_IP=192.168.1.42 \
  nav-robot
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PC / Laptop  (TCP SERVER – runs heavy AI)                      │
│                                                                 │
│  main_server.py              ← entry point                      │
│  ├── V-JEPA 2                ← future-scene prediction          │
│  ├── SSv2 temporal rules     ← motion pattern classification    │
│  ├── Decision fuser          ← weighted risk fusion + hysteresis│
│  ├── Visualization (OpenCV)  ← annotated HUD overlay            │
│  └── TCP servers                                                │
│      ├── ports 5003/8003     ← operator UI viewer (ai_viewer)   │
│      └── ports 5004/8004     ← robot (Pi) connection            │
└───────────────────┬──────────────────────────────────────────── ┘
                    │  CMD_AIMOVE / CMD_STOP ↓  ↑ CMD_DETECTION
                    │  CMD_MOTOR (manual)    ↓  ↑ JPEG frames
┌───────────────────┴──────────────────────────────────────────── ┐
│  Raspberry Pi  (TCP CLIENT – hardware + fast AI)                │
│                                                                 │
│  main_robot.py               ← connects to PC server            │
│  ├── YOLOv8n                 ← instantaneous obstacle detection  │
│  ├── picamera2               ← JPEG camera stream → port 8004   │
│  ├── tankMotor (gpiozero)    ← executes CMD_AIMOVE / CMD_MOTOR  │
│  └── Ultrasonic sensor       ← distance included in CMD_DETECTION│
└─────────────────────────────────────────────────────────────────┘
                    ▲
                    │  CMD_AISTATUS (live AI state)
                    │  annotated JPEG frames
┌───────────────────┴──────────────────────────────────────────── ┐
│  Operator laptop  (UI viewer)                                   │
│                                                                 │
│  Code/Client/ai_viewer.py    ← connects to PC server 5003/8003  │
│  Shows: action, risk bars, V-JEPA 2 label, motion pattern       │
│  AUTO mode:   AI decision fuser drives the robot                │
│  MANUAL mode: operator drives via buttons / arrow keys          │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- The PC is the TCP *server* (binds and listens); the robot and UI viewer are
  TCP *clients* (connect outbound to the PC).
- **YOLOv8n runs on the Pi** (fast reactive detection, low latency).
- **V-JEPA 2 + SSv2 + decision run on the PC** (GPU-capable heavy inference).
- Only compact detection results (`CMD_DETECTION`) are sent over the network —
  not raw feature tensors.

---

## What each AI model does

| Model | Nickname | What it does | Where it runs |
|---|---|---|---|
| **YOLOv8n** | "The Photographer" | Spots obstacles in the current frame; sends aggregated risk+position to PC | Pi |
| **V-JEPA 2** | "The Fortune Teller" | Predicts what the scene will look like 0.5 s from now in latent space | PC |
| **SSv2 temporal rules** | "The Behaviour Analyst" | Classifies the obstacle's motion pattern (APPROACHING / CROSSING / BLOCKING …) | PC |
| **Decision fuser** | "The Judge" | Combines all three risk signals into one action (FORWARD / SLOW / STOP / REROUTE) | PC |

### V-JEPA 2 anchor embeddings

V-JEPA 2 works by predicting **future latent embeddings** (not pixels). To turn
those embeddings into a risk score the system compares them to two reference
points in embedding space:

- `obstacle_anchor` — average embedding of corridor frames that contain a
  centred blocker
- `clear_anchor`    — average embedding of obstacle-free corridor frames

The cosine similarity difference becomes `predicted_risk ∈ [0, 1]`.

---

## How V-JEPA 2 improves navigation

A person entering at the frame edge has **low** current detection risk (small
bounding box, off-centre) but produces a **predicted** embedding much closer to
the obstacle anchor than the clear anchor.  The robot begins decelerating several
frames before the baseline (YOLO-only) system reacts.

Predictive early-warning path in `Code/Server/decision.py`:
```python
if self._mode == "predictive" and world_model_label == "BLOCKED" and action == Action.FORWARD:
    action = Action.SLOW   # decelerate proactively
```

---

## Baseline vs Predictive comparison

| Feature | Baseline | Predictive |
|---|---|---|
| YOLOv8 detection (on Pi) | ✓ | ✓ |
| V-JEPA 2 future prediction | ✗ (weight = 0) | ✓ (weight = 0.45) |
| SSv2 temporal patterns | ½ weight | full weight |
| Ultrasonic guard | ✓ | ✓ |
| V-JEPA 2 early-warning deceleration | ✗ | ✓ |

Both modes run on the **same code path** — only the weight vector changes.

---

## Project structure

```
Robot-navigation-world-model-demo/
├── Code/
│   ├── Server/          ← PC AI server (V-JEPA 2 + SSv2 + decision + TCP)
│   │   ├── main_server.py        ← PC entry point
│   │   ├── robot_connection.py   ← accepts robot TCP connection; parses CMD_DETECTION
│   │   ├── ai_pipeline.py        ← AI orchestration loop
│   │   ├── camera_buffer.py      ← rolling frame buffer (demo / live / tcp modes)
│   │   ├── detector.py           ← YOLOv8n (demo mode only; live mode uses Pi)
│   │   ├── world_model.py        ← V-JEPA 2
│   │   ├── temporal_action.py    ← SSv2-style motion patterns
│   │   ├── decision.py           ← risk fusion + hysteresis
│   │   ├── robot_control.py      ← motor controller (real / mock / TCP)
│   │   ├── visualization.py      ← OpenCV HUD overlay
│   │   ├── ai_logger.py          ← CSV + annotated JPEG archive
│   │   └── config.yaml           ← PC-side configuration
│   ├── Robot/           ← Raspberry Pi client (hardware + YOLOv8n)
│   │   ├── main_robot.py         ← Pi entry point
│   │   ├── detector_robot.py     ← YOLOv8n wrapper (runs on Pi)
│   │   ├── tcp_robot_client.py   ← outbound TCP client to PC
│   │   ├── camera.py             ← picamera2 streaming
│   │   ├── motor.py              ← tankMotor (gpiozero)
│   │   ├── ultrasonic.py         ← distance sensor
│   │   ├── parameter.py          ← Pi hardware version detection
│   │   ├── requirements_robot.txt← Pi Python deps (includes ultralytics)
│   │   └── config_robot.yaml     ← Pi-side configuration
│   └── Client/          ← UI viewer (connects to PC)
│       └── ai_viewer.py          ← PyQt5 operator display (AUTO + MANUAL modes)
├── tests_rpi/           ← unit tests (no GPU / hardware required)
├── Dockerfile.server    ← PC Docker image (V-JEPA 2 + SSv2 + decision)
├── Dockerfile.robot     ← Pi Docker image (arm64; YOLOv8n + hardware)
├── docker-compose.server.yml
├── docker-compose.robot.yml
├── requirements_server.txt  ← PC Python deps
└── assets/demo_clips/       ← corridor video for demo mode
```

---

## TCP protocol

| Command | Direction | Format | Meaning |
|---|---|---|---|
| `CMD_DETECTION` | Pi → PC | `CMD_DETECTION#<risk_pct>#<in_center>#<area_pct>#<cx_pct>#<sonic_cm>` | YOLOv8 result + ultrasonic (main Pi→PC message) |
| `CMD_AIMOVE` | PC → Pi | `CMD_AIMOVE#<FORWARD\|SLOW\|STOP\|REROUTE>` | AI-computed action; Pi maps to motor PWM |
| `CMD_MOTOR` | UI → PC → Pi | `CMD_MOTOR#<L>#<R>` | Manual motor command relayed through PC |
| `CMD_STOP` | PC → Pi | `CMD_STOP` | Emergency halt (hard safety) |
| `CMD_KILL` | PC → Pi | `CMD_KILL` | Shutdown robot process |
| `CMD_AIMODE` | UI → PC | `CMD_AIMODE#<0/1/2>` | Mode change from operator |
| `CMD_KILL` | UI → PC | `CMD_KILL#0` | Shutdown from operator |
| `CMD_AISTATUS` | PC → UI | `CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>` | Live AI state |
| Video frames | Pi → PC | 4-byte LE uint32 length + JPEG | Camera stream for V-JEPA 2 (port 8004) |
| Video frames | PC → UI | 4-byte LE uint32 length + JPEG | Annotated frames (port 8003) |

---

## Setup

### PC / Laptop (TCP server – binds ports 5003/5004/8003/8004)

```bash
# Clone the repo
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# Install Python packages
pip install -r requirements_server.txt

# V-JEPA 2 weights (~300 MB) are downloaded from HuggingFace on first run.
# If offline, the system falls back to a lightweight stub encoder.
```

### Raspberry Pi (TCP client – robot hardware + YOLOv8n)

The Pi connects **outbound** to the PC server; it does not bind any ports.

```bash
# System packages (run once on the Pi)
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-libcamera \
  python3-gpiozero python3-kms++ python3-prctl libatlas-base-dev

# Python packages (includes ultralytics for YOLOv8n)
cd /path/to/robot-navigation-world-model-demo
pip3 install -r Code/Robot/requirements_robot.txt

# YOLOv8n weights (~6 MB) download automatically on first run.
# To pre-download:
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## How to run

### Option A – Demo mode (no robot hardware needed)

**Step 1** – Place a corridor video clip at:
```
assets/demo_clips/corridor.mp4
```
Any short indoor walkway video works (30 s is plenty).

**Step 2** – Start the PC server in demo mode:

```bash
cd Code/Server

# Predictive mode (V-JEPA 2 active)
python main_server.py --mode demo --nav predictive

# Baseline reactive mode (for comparison)
python main_server.py --mode demo --nav baseline

# Custom video file
python main_server.py --mode demo --video /path/to/my_video.mp4

# Headless (no OpenCV window)
python main_server.py --mode demo --no-display
```

In demo mode the server runs local YOLOv8n (no Pi needed).

**Step 3 (optional)** – Connect the UI viewer while the server runs:

```bash
cd Code/Client
python ai_viewer.py
# Type 127.0.0.1 (localhost) in the IP field and click Connect
```

---

### Option B – Live mode (physical Freenove Tank Robot)

**Step 1** – Find the PC's IP address:
```bash
ip route get 1 | awk '{print $7; exit}'
```

**Step 2** – Edit the robot config to point at your PC:
```yaml
# Code/Robot/config_robot.yaml
server_ip: "192.168.1.42"   # ← your PC IP here
```

**Step 3** – Start the PC server:
```bash
cd Code/Server
python main_server.py --mode live --nav predictive
```
The server prints its ports and waits for the robot to connect.

**Step 4** – On the Raspberry Pi, connect the robot client:
```bash
cd Code/Robot
python main_robot.py --server-ip 192.168.1.42
```
The robot:
1. Loads YOLOv8n locally
2. Starts streaming JPEG frames to PC (for V-JEPA 2)
3. Sends `CMD_DETECTION` (YOLO result + ultrasonic) to PC every 100 ms
4. Awaits `CMD_AIMOVE` (AI actions) and `CMD_MOTOR` (manual commands)

**Step 5 (optional)** – Connect the UI viewer from any machine on the network:
```bash
cd Code/Client
python ai_viewer.py
# Enter the PC's IP address (192.168.1.42) and click Connect
```

---

### Option C – Docker (recommended for reproducibility)

**PC server (`Dockerfile.server`):**

The server image is built on `python:3.11-slim` (multi-platform: arm64 + amd64).
No CUDA required. `opencv-python-headless` is used so no GL/X11 display libraries
are needed — the server always runs headless (`--no-display`).

```bash
# Build (Mac Apple Silicon, Linux amd64, Linux arm64 — all work)
docker build -f Dockerfile.server -t nav-server .

# Demo mode — place a video at assets/demo_clips/corridor.mp4 first
docker run --rm \
  -p 5003:5003 -p 8003:8003 \
  -p 5004:5004 -p 8004:8004 \
  -v "$(pwd)/assets:/app/assets:ro" \
  -v "$(pwd)/logs_rpi:/app/logs_rpi" \
  nav-server

# Live mode — server waits for Pi to connect
docker run --rm \
  -p 5003:5003 -p 8003:8003 \
  -p 5004:5004 -p 8004:8004 \
  nav-server python main_server.py --mode live --nav predictive --no-display

# Baseline comparison
docker run --rm \
  -p 5003:5003 -p 8003:8003 \
  -p 5004:5004 -p 8004:8004 \
  -v "$(pwd)/assets:/app/assets:ro" \
  nav-server python main_server.py --mode demo --nav baseline --no-display

# Via docker compose (sets NAV_MODE and NAV_STRATEGY env vars)
NAV_MODE=live NAV_STRATEGY=predictive docker compose -f docker-compose.server.yml up

# NVIDIA GPU (Linux only — requires nvidia-container-toolkit)
docker build \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
  -f Dockerfile.server -t nav-server-gpu .
docker run --rm --gpus all \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 \
  nav-server-gpu python main_server.py --mode live --nav predictive --no-display
```

**Raspberry Pi robot (`Dockerfile.robot`, arm64):**

The robot image uses `python:3.11-slim-bookworm`.  Camera/GPIO packages
(`python3-picamera2`, `python3-libcamera`, `python3-kms++`) are Raspberry Pi OS
specific and **must be installed on the Pi host** before running the container —
the container accesses them via device passthrough:

```bash
# One-time Pi host setup (run directly on the Pi, not in Docker)
sudo apt-get install -y python3-picamera2 python3-libcamera python3-kms++

# Build (cross-compile on Mac/Linux, or build directly on the Pi)
docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .

# Run on Pi — replace 192.168.1.42 with your PC's IP
docker run --rm --privileged \
  --device /dev/video0:/dev/video0 \
  --device /dev/gpiochip4:/dev/gpiochip4 \
  -e SERVER_IP=192.168.1.42 \
  nav-robot

# Pi 4 uses /dev/gpiochip0 instead of /dev/gpiochip4
docker run --rm --privileged \
  --device /dev/video0:/dev/video0 \
  --device /dev/gpiochip0:/dev/gpiochip0 \
  -e SERVER_IP=192.168.1.42 \
  nav-robot

# Via docker compose
SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up
```

> **YOLOv8n weights** (~6 MB) are pre-downloaded during `docker build` so the
> container starts immediately without a network call.

---

## Operator UI controls

| Control | Action |
|---|---|
| **AUTO MODE** / `Ctrl+A` | AI decision fuser drives the robot (predictive mode) |
| **MANUAL MODE** / `Ctrl+M` | Operator controls motors directly; AI is paused |
| **PREDICTIVE** / `Ctrl+P` | Switch to V-JEPA 2 predictive mode (AUTO only) |
| **BASELINE** / `Ctrl+B` | Switch to reactive-only baseline mode (AUTO only) |
| **Arrow keys** ↑↓←→ | Drive in MANUAL mode (hold = move, release = stop) |
| **On-screen drive buttons** | Same as arrow keys; tap-and-hold |
| **Speed: Full / Slow** | Toggle between full PWM and slow PWM in MANUAL mode |
| **EMERGENCY STOP** / `Space` / `Esc` | Immediately cut motor power and disable AI |
| **SHUTDOWN SERVER** / `Ctrl+Q` | Send `CMD_KILL` – stop motors, shut down server |

---

## Calibrate V-JEPA 2 anchors for your corridor

The default anchors work for most indoor corridors.  To recalibrate:

```bash
cd Code/Server
python main_server.py --build-anchors
# 'o' → label frame as obstacle
# 'c' → label frame as clear
# 'q' → save and exit
```

At least 10 frames per class is recommended.

---

## Configuration

### PC server (`Code/Server/config.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `navigation_mode` | `predictive` | Starting navigation mode |
| `world_model.risk_similarity_threshold` | `0.55` | V-JEPA 2 BLOCKED sensitivity |
| `decision.weights.world_model` | `0.45` | V-JEPA 2 contribution to fused risk |
| `decision.low_risk_max` | `0.30` | Below this → FORWARD |
| `decision.medium_risk_max` | `0.60` | Below this → SLOW, above → STOP/REROUTE |
| `robot.ultrasonic_stop_cm` | `15.0` | Hard stop distance (cm) from robot |
| `world_model.run_every_n_frames` | `8` | V-JEPA 2 cadence (CPU saving) |
| `server.cmd_port` | `5003` | UI viewer command port |
| `server.video_port` | `8003` | UI viewer video port |
| `server.robot_cmd_port` | `5004` | Robot command port |
| `server.robot_video_port` | `8004` | Robot video port |

### Pi robot (`Code/Robot/config_robot.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `server_ip` | `192.168.1.100` | PC server IP address |
| `detector.model` | `yolov8n.pt` | YOLOv8 model (runs locally on Pi) |
| `detector.conf` | `0.35` | Detection confidence threshold |
| `detector.run_every_n_frames` | `2` | YOLOv8 cadence (CPU saving) |
| `detector.center_zone_width` | `0.40` | Fraction of frame = centre danger zone |
| `camera.stream_width/height` | `400 × 300` | Camera JPEG resolution |
| `robot.speed_full` | `1500` | Full-speed PWM |
| `robot.speed_slow` | `800` | Reduced-speed PWM |
| `ultrasonic.read_interval` | `0.1` | Seconds between CMD_DETECTION sends |

---

## Running tests

```bash
# Tests that need no GPU or hardware (run on any machine)
pytest tests_rpi/ -v

# Original Freenove tests
pytest tests/ -v
```

Tests cover:
- Decision fusion and hysteresis
- SSv2 temporal motion patterns
- V-JEPA 2 mathematics (no GPU needed)
- Camera buffer (demo + tcp modes)
- Robot TCP client protocol (including `send_detection`)
- Robot connection server protocol (including CMD_DETECTION parsing and `send_aimove`)

---

## Logging

Each run creates a timestamped directory under `logs_rpi/`:

```
logs_rpi/
└── run_20240515_143022_predictive/
    ├── navigation_log.csv   # one row per frame
    ├── system.log           # Python logging output
    └── frames/              # annotated JPEGs (every 5th frame)
```

The CSV captures all three risk signals separately (`detector_risk`,
`world_model_risk`, `temporal_risk`) so you can plot predictive vs baseline risk
trajectories from the same scenario.

---

## Success criteria

- Robot reaches the goal point reliably in both modes
- Predictive mode visibly begins decelerating **earlier** than baseline mode
- The V-JEPA 2 label shows `BLOCKED` before the YOLOv8 detector fills the risk bar
- Motion is smoother in predictive mode (fewer full stops from a cold start)
- System runs stably at ≥ 8 FPS in demo mode
- All signals are logged to CSV for post-run analysis
