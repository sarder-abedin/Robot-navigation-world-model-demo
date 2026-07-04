# Freenove Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot Kit for
Raspberry Pi (FNK0077)** that uses **V-JEPA 2** as a world model to anticipate
future obstacles — not just react to what is currently visible.

---

## Quick Start

> **Rebuild after pulling changes** — the server and robot are **two separate
> images**. After changing code (or `git pull`), rebuild **both** (use the same
> flags as their build commands below) or the stale one keeps its old behaviour:
>
> ```bash
> docker build --no-cache -f Dockerfile.server -t nav-server .
> docker build --no-cache --platform linux/arm64 -f Dockerfile.robot -t nav-robot .
> ```
>
> `--no-cache` forces a fresh build (skips Docker's layer cache); `--platform
> linux/arm64` targets the Pi (drop it if you build on the Pi itself). A robot log
> line like `Camera started via picamera2` instead of `Camera streaming via
> picamera2 JpegEncoder` means the robot image is stale.

### Fastest path — Docker on Mac / Linux (no robot needed)

```bash
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# Build the server image (arm64 Mac Apple Silicon + amd64 Linux both work)
docker build -f Dockerfile.server -t nav-server .

# Demo mode — no robot needed; supply any corridor video first:
#   mkdir -p assets/demo_clips && cp /path/to/corridor.mp4 assets/demo_clips/
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 -v "$(pwd)/assets:/app/assets:ro" -v "$(pwd)/logs_rpi:/app/logs_rpi" nav-server
# Wait for: "[start_server] Starting Streamlit on http://0.0.0.0:8501"
# Then open http://localhost:8501 → enter "localhost" as server IP → Connect

# Live mode — server + Streamlit UI, waits for Pi to connect on ports 5004/8004
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 -e NAV_MODE=live nav-server
# Then open http://localhost:8501 → enter "localhost" as server IP → Connect
```

> **V-JEPA 2 weights** (~300 MB) are downloaded from HuggingFace automatically
> on first run.  No GPU required — CPU-only inference works out of the box.
>
> **Hardware acceleration** — both heavy models (V-JEPA 2 and SSv2) default to
> `device: auto`, which picks the best available at load: **CUDA → MPS → CPU**.
> On an NVIDIA host (incl. DGX) run the GPU image with `--gpus all` (see the GPU
> build below) and they use CUDA; SSv2 also classifies ~2× more often on a GPU.
> On a **native** Mac they use Apple **MPS**. Note that a Docker container **on a
> Mac** has no Metal passthrough, so it stays on **CPU** even with unified memory —
> for GPU on a Mac, run the server natively (`python main_server.py …`). Force a
> device by setting `world_model.device` / `ssv2.device` to `cuda`/`mps`/`cpu`.

### Fastest path — Raspberry Pi robot (Docker)

```bash
# Build the robot image (cross-compile on Mac/Linux or build directly on Pi)
docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .

# Recommended: use compose — it already wires up the camera (video0 + media0/1),
# both gpiochip nodes, and the udev mount. Just set your PC's IP:
SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up
# If you get "unknown shorthand flag: 'f' in -f", the Compose V2 plugin is not
# installed; use the hyphenated V1 binary instead, or the plain `docker run` below:
#   SERVER_IP=192.168.1.42 docker-compose -f docker-compose.robot.yml up

# Or the equivalent single docker run (Pi 5) — replace 192.168.1.42 with your PC's IP.
# Note BOTH gpiochip mappings: on Pi 5 the lgpio pin factory needs the RP1
# controller visible at chip 4 as well, or motors fail with "can not open gpiochip".
docker run --rm --privileged --device /dev/video0:/dev/video0 --device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4 -v /run/udev:/run/udev:ro -e SERVER_IP=192.168.1.42 nav-robot
```

> **Camera (CSI)** — the container uses **picamera2/libcamera** to drive the Pi CSI camera (same stack as Freenove). `-v /run/udev:/run/udev:ro` is **required** so libcamera can enumerate the camera inside the container; with `--privileged` the camera device nodes under `/dev` are already available. If picamera2 cannot be imported the code falls back to OpenCV V4L2 on `/dev/video0`, but the CSI camera generally does **not** produce frames through that path — so if the PC log says *"waiting for camera frames"*, confirm the udev mount is present.
>
> **Camera orientation** — the feed is streamed upright by default (no flip), so the UI, V-JEPA 2 and YOLO all get the correctly-oriented frame. If your camera is mounted **inverted** and the image looks upside-down, flip it with `-e CAMERA_HFLIP=1 -e CAMERA_VFLIP=1` (no rebuild needed) or set `camera.hflip`/`camera.vflip: true` in `config_robot.yaml`.
>
> **GPIO chip** — on Pi 5 the RP1 controller is `/dev/gpiochip0`, but the lgpio pin factory also probes `gpiochip4`, so map the host controller to **both** container nodes (`--device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4`) as shown above. If motors log `can not open gpiochip`, that second mapping is missing. Run `gpiodetect` on the host to confirm which chip is `pinctrl-rp1`.
>
> **Motor speed** — the robot drives **slowly** by default (`speed_full: 1600`, `speed_slow: 1000` out of 4095) so it stays reactive to the CPU pipeline. Tune it **without rebuilding** via `-e SPEED_FULL=<n> -e SPEED_SLOW=<n>` (or edit `config_robot.yaml`): raise it a little if the robot doesn't start moving, lower it if it's still too fast. A soft-start ramp (`robot.soft_start`) blunts the current spike so the Pi doesn't brown out on drive.
>
> ⚠️ **Compose forwards env vars only if they're declared.** With `docker compose`, an inline var like `SPEED_FULL=1500` reaches the container **only** because `docker-compose.robot.yml` lists it under `environment:`. The forwarded vars are: `SERVER_IP`, `SPEED_FULL`, `SPEED_SLOW`, `CAMERA_HFLIP`, `CAMERA_VFLIP`, `GPIO_CHIP`. Example: `SERVER_IP=192.168.68.107 SPEED_FULL=1500 docker compose -f docker-compose.robot.yml up`.
>
> ⚠️ **`docker compose up` reuses stale images.** Both compose files pin an
> `image:` name, so `up` alone will **not** rebuild after you change code or
> `git pull` — it silently reruns the old image and your new flags appear to "do
> nothing". Always pass `--build` (or run `docker compose ... build` first). Build
> the **robot** image **on the Pi** (arm64 + Raspberry Pi apt repos); it will not
> build on a Mac/PC.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PC / Laptop  (TCP SERVER – runs ALL AI)    [venv OR Docker]    │
│                                                                 │
│  main_server.py              ← entry point                      │
│  ├── YOLOv8n                 ← object detection (all modes)      │
│  ├── V-JEPA 2                ← future-scene prediction          │
│  ├── SSv2 (VideoMAE)         ← genuine action recognition       │
│  ├── Decision fuser          ← weighted risk fusion + hysteresis│
│  ├── Visualization (OpenCV)  ← annotated HUD overlay            │
│  └── TCP servers                                                │
│      ├── ports 5003/8003     ← operator UI viewer               │
│      └── ports 5004/8004     ← robot (Pi) connection            │
└───────────────────┬──────────────────────────────────────────── ┘
                    │  CMD_AIMOVE / CMD_MOTOR ↓  ↑ CMD_SONIC + JPEG frames
┌───────────────────┴──────────────────────────────────────────── ┐
│  Raspberry Pi  (TCP CLIENT – thin client, no AI)  [Docker]     │
│                                                                 │
│  main_robot.py               ← connects to PC server            │
│  ├── picamera2               ← JPEG camera stream → port 8004   │
│  ├── tankMotor (gpiozero)    ← executes CMD_AIMOVE / CMD_MOTOR  │
│  └── Ultrasonic sensor       ← local hard-stop safety →         │
│                                CMD_SONIC                        │
└─────────────────────────────────────────────────────────────────┘
                    ▲
                    │  CMD_AISTATUS (live AI state)
                    │  annotated JPEG frames
┌───────────────────┴──────────────────────────────────────────── ┐
│  Operator browser  (Streamlit UI viewer)                        │
│                                                                 │
│  Code/Client/streamlit_viewer.py  ← served on port 8501        │
│    Open http://localhost:8501 — no install needed on Mac/Linux  │
│  Code/Client/ai_viewer.py         ← PyQt5 alternative (native) │
│  Shows: action, risk bars, V-JEPA 2 label, motion pattern       │
│  AUTO mode:   AI decision fuser drives the robot                │
│  MANUAL mode: operator drives via on-screen buttons             │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- The PC is the TCP *server* (binds and listens); the robot and UI viewer are
  TCP *clients* (connect outbound to the PC).
- **All AI runs on the PC** — YOLOv8n object detection, V-JEPA 2, SSv2 and the
  decision fuser (GPU-capable heavy inference). The Pi runs no AI.
- The Pi is a **thin client**: it streams JPEG camera frames to the PC and sends
  `CMD_SONIC` (its ultrasonic reading, the local hard-stop safety); the PC runs
  YOLO on those frames.

---

## What each AI model does

| Model | Nickname | What it does | Where it runs |
|---|---|---|---|
| **YOLOv8n** | "The Photographer" | Spots obstacles in the current frame; produces aggregated risk+position **and the largest obstacle's class label** | PC |
| **V-JEPA 2** | "The Fortune Teller" | Predicts what the scene will look like 0.5 s from now in latent space | PC |
| **SSv2 temporal rules** | "The Behaviour Analyst" | Classifies the obstacle's motion pattern (APPROACHING / CROSSING / BLOCKING …) — drives `temporal_risk` | PC |
| **SSv2 model (VideoMAE)** | "The Narrator" | A **real** Something-Something-V2 video classifier; its "something" slot is filled with YOLO's object → e.g. *"person moving closer"*. Annotation/log only | PC |
| **Decision fuser** | "The Judge" | Combines all three risk signals into one action (FORWARD / SLOW / STOP / REROUTE) | PC |

### Genuine SSv2 action recognition (YOLO-filled)

`Code/Server/ssv2_model.py` runs a **real** SSv2-finetuned video classifier
(VideoMAE, `MCG-NJU/videomae-base-finetuned-ssv2`) over the rolling clip. SSv2
labels are templated phrases with a *"something"* placeholder — we fill that slot
with the **largest obstacle's YOLO class** (from the PC's own YOLO detection),
producing a human sentence like **"person moving closer"** or
**"chair pushed from left to right"**.

- It is **annotation + logging only** — it does **not** drive navigation (the
  fast heuristic in `temporal_action.py` still supplies `temporal_risk`), so
  navigation behaviour is unchanged.
- Shown on the video HUD (`SSv2: …`) and as an `SSv2:` line in the AI-state panel,
  and written to the CSV log.
- Runs every `ssv2.run_every_n_frames` (default 16) on CPU, auto-halved on a GPU;
  uses `device: auto` (CUDA → MPS → CPU). First run downloads the checkpoint
  (~350 MB) from HuggingFace (`transformers` is already a dependency for V-JEPA 2).
  If the model/weights are unavailable it falls back to a stub that still fills
  the object. Tune or disable it in `config.yaml` (`ssv2.enabled: false`) if a
  CPU-only host can't run two video transformers.

### Run logging (PC-side, operator-controlled)

Run logging (CSV + annotated frames) is written **entirely on the PC server**
(`logs_rpi/…`) — the robot never logs. It is **off by default** and controlled by
the operator:

- **Before the run:** `--logging on` (or `docker run -e NAV_LOGGING=1 …`).
- **During the run:** the **"Run Logging" toggle** in the Streamlit UI (sends
  `CMD_LOGGING#<0|1>`).

The CSV includes an `ssv2` column with the composed sentence.

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
| YOLOv8 detection (on PC) | ✓ | ✓ |
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
│   │   ├── robot_connection.py   ← accepts robot TCP connection; parses CMD_SONIC
│   │   ├── ai_pipeline.py        ← AI orchestration loop
│   │   ├── camera_buffer.py      ← rolling frame buffer (demo / live / tcp modes)
│   │   ├── detector.py           ← YOLOv8n (runs on the PC in all modes)
│   │   ├── world_model.py        ← V-JEPA 2
│   │   ├── temporal_action.py    ← SSv2-style motion-pattern heuristic (drives temporal_risk)
│   │   ├── ssv2_model.py         ← genuine SSv2 model (VideoMAE); YOLO-filled sentence for annotation/log
│   │   ├── decision.py           ← risk fusion + hysteresis
│   │   ├── robot_control.py      ← motor controller (real / mock / TCP)
│   │   ├── visualization.py      ← OpenCV HUD overlay
│   │   ├── ai_logger.py          ← CSV + annotated JPEG archive
│   │   └── config.yaml           ← PC-side configuration
│   ├── Robot/           ← Raspberry Pi client (thin client: hardware only, no AI)
│   │   ├── main_robot.py         ← Pi entry point
│   │   ├── tcp_robot_client.py   ← outbound TCP client to PC
│   │   ├── camera.py             ← picamera2 streaming
│   │   ├── motor.py              ← tankMotor (gpiozero)
│   │   ├── ultrasonic.py         ← distance sensor
│   │   ├── parameter.py          ← Pi hardware version detection
│   │   ├── requirements_robot.txt← Pi Python deps (lightweight, no torch/ultralytics)
│   │   └── config_robot.yaml     ← Pi-side configuration
│   └── Client/          ← UI viewers (connect to PC)
│       ├── streamlit_viewer.py   ← browser UI (port 8501, bundled in Docker)
│       ├── desktop_viewer.py     ← native desktop window wrapping Streamlit (pywebview)
│       └── ai_viewer.py          ← PyQt5 desktop UI (run natively if preferred)
├── tests_rpi/           ← unit tests (no GPU / hardware required)
├── Dockerfile.server    ← PC Docker image (V-JEPA 2 + SSv2 + decision)
├── Dockerfile.robot     ← Pi Docker image (arm64; lightweight, hardware only)
├── docker-compose.server.yml
├── docker-compose.robot.yml
├── requirements_server.txt  ← PC Python deps
└── assets/demo_clips/       ← corridor video for demo mode
```

---

## TCP protocol

| Command | Direction | Format | Meaning |
|---|---|---|---|
| `CMD_SONIC` | Pi → PC | `CMD_SONIC#<sonic_cm>` | Ultrasonic distance (the Pi's local hard-stop safety); the Pi runs no detection |
| `CMD_AIMOVE` | PC → Pi | `CMD_AIMOVE#<FORWARD\|SLOW\|STOP\|REROUTE>` | AI-computed action; Pi maps to motor PWM |
| `CMD_MOTOR` | UI → PC → Pi | `CMD_MOTOR#<L>#<R>` | Manual motor command relayed through PC |
| `CMD_STOP` | PC → Pi | `CMD_STOP` | Emergency halt (hard safety) |
| `CMD_KILL` | PC → Pi | `CMD_KILL` | Shutdown robot process |
| `CMD_AIMODE` | UI → PC | `CMD_AIMODE#<0/1/2>` | Mode change from operator |
| `CMD_LOGGING` | UI → PC | `CMD_LOGGING#<0/1>` | Toggle PC-side run logging |
| `CMD_KILL` | UI → PC | `CMD_KILL#0` | Shutdown from operator |
| `CMD_AISTATUS` | PC → UI | `CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>#<ssv2_sentence>` | Live AI state (SSv2 sentence is the last, optional field) |
| Video frames | Pi → PC | 4-byte LE uint32 length + JPEG | Camera stream for V-JEPA 2 (port 8004) |
| Video frames | PC → UI | 4-byte LE uint32 length + JPEG | Annotated frames (port 8003) |

---

## Setup

### PC / Laptop (TCP server – binds ports 5003/5004/8003/8004)

The server runs **two ways** — pick one. Use the **virtualenv** path for development
and GPU on a Mac (MPS); use **Docker** for a one-command reproducible run. Both are
documented step by step under [How to run](#how-to-run).

**Option 1 — Python virtualenv (native):**

```bash
# Clone the repo
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install server + UI dependencies into the venv
pip install --upgrade pip
pip install -r requirements_server.txt

# V-JEPA 2 weights (~300 MB) are downloaded from HuggingFace on first run.
# If offline, the system falls back to a lightweight stub encoder.
```

**Option 2 — Docker:** no local Python needed; see
[Option C – Docker](#option-c--docker-recommended-for-reproducibility).

### Raspberry Pi (TCP client – robot hardware) — **Docker-based**

The Pi runs **in Docker** (the supported path) as a **thin client** that runs no
AI: the lightweight image bundles picamera2, libcamera and GPIO (no
torch/ultralytics), all pinned to versions that coexist on Pi OS. It connects
**outbound** to the PC server; it binds no ports.

```bash
# On the Pi, from the repo root:
docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .
# (drop --platform if building on the Pi itself). Full run commands under
# "How to run → Option C".
```

<details><summary>Bare-metal Pi install (unsupported fallback, no Docker)</summary>

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-libcamera python3-gpiozero python3-kms++ python3-prctl libatlas-base-dev
cd /path/to/robot-navigation-world-model-demo
python3 -m venv .venv --system-site-packages   # picamera2/libcamera come from apt
source .venv/bin/activate
pip3 install -r Code/Robot/requirements_robot.txt
```
</details>

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

**Step 3 (optional)** – Open the operator UI viewer.

When the server runs **natively** (not Docker) the viewer is launched separately.
There are **three ways** to open the *same* UI — pick one. All run in a second
terminal while `main_server.py` runs in the first, and in every case you enter
the server IP (`127.0.0.1` if it's the same machine) and click **Connect**.

First, one-time, install the viewer dependencies (no torch needed — the UI runs
no AI):

```bash
pip install -r requirements_client.txt
```

**Option 1 — Browser tab (simplest):**

```bash
streamlit run Code/Client/streamlit_viewer.py
# Opens http://localhost:8501 in your default browser
```

**Option 2 — Native desktop window (app-like, no browser tab):**

```bash
cd Code/Client
python desktop_viewer.py
# Opens the SAME Streamlit UI inside a native OS window (via pywebview).
#   • macOS / Windows: works out of the box.
#   • Linux: also install a webview backend once:
#       sudo apt-get install -y python3-gi gir1.2-webkit2-4.1
# Options: python desktop_viewer.py --port 8600 --title "Nav" --width 1400 --height 900
```

`desktop_viewer.py` starts Streamlit headless, waits for it, opens the window,
and stops Streamlit when you close the window — so it's a single command.

**Option 3 — PyQt5 desktop viewer (alternative native UI, keyboard shortcuts):**

```bash
cd Code/Client
python ai_viewer.py
```

> All three show the same live video, risk bars, V-JEPA 2 / SSv2 labels and
> AUTO/MANUAL controls. Inside Docker the **browser** viewer starts automatically
> (a native window needs a display, so `desktop_viewer.py` / `ai_viewer.py` are
> native-only).

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

**Step 4** – On the Raspberry Pi, start the robot client **in Docker** (build once,
then run — see [Option C](#option-c--docker-recommended-for-reproducibility) for the
full flags):
```bash
# On the Pi:
SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up --build
```
The robot:
1. Streams JPEG frames to the PC (the PC runs YOLOv8n + V-JEPA 2 on them)
2. Reads the ultrasonic sensor and sends `CMD_SONIC#<cm>` (its local hard-stop
   safety) — it runs **no** AI itself
3. Awaits `CMD_AIMOVE` (AI actions) and `CMD_MOTOR` (manual commands) and drives
   the motors

**Step 5 (optional)** – Open the browser UI viewer from any machine on the network:

```bash
# Option A – Streamlit (browser, recommended)
pip install streamlit   # one-time
streamlit run Code/Client/streamlit_viewer.py
# Opens http://localhost:8501 — enter the PC server IP and click Connect

# Option B – PyQt5 desktop viewer
pip install PyQt5 opencv-python numpy   # one-time
cd Code/Client
python ai_viewer.py
# Enter the PC's IP address (e.g. 192.168.1.42) and click Connect
```

| Where viewer runs | Server IP to enter |
|---|---|
| Same machine as the server | `localhost` or `127.0.0.1` |
| Different machine on LAN | PC's LAN IP (e.g. `192.168.1.42`) |
| Mac with server in Docker Desktop | `localhost` (Docker Desktop maps ports to host) |

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
# The container starts BOTH the AI server AND the Streamlit browser UI.
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 -v "$(pwd)/assets:/app/assets:ro" -v "$(pwd)/logs_rpi:/app/logs_rpi" nav-server
# Then open http://localhost:8501 — enter "localhost" as the server IP.

# Live mode — server + viewer, waiting for Pi to connect
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 -e NAV_MODE=live nav-server

# Server only (no Streamlit viewer)
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 nav-server python main_server.py --mode demo --nav predictive --no-display

# Via docker compose (recommended — forwards NAV_MODE, NAV_STRATEGY).
# Pass --build the first time and after every git pull (compose reuses stale images).
NAV_MODE=live NAV_STRATEGY=predictive docker compose -f docker-compose.server.yml up --build

# NVIDIA GPU (Linux only — requires nvidia-container-toolkit)
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 -f Dockerfile.server -t nav-server-gpu .
docker run --rm --gpus all -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 nav-server-gpu
```

**Raspberry Pi robot (`Dockerfile.robot`, arm64):**

The robot image uses `python:3.11-slim-bookworm`.

- **Camera (CSI)** — uses `picamera2`/`libcamera` (the same stack Freenove uses). Mount `-v /run/udev:/run/udev:ro` so libcamera can enumerate the camera inside the container. It falls back to OpenCV V4L2 on `/dev/video0` only if picamera2 cannot be imported, but the CSI camera generally will not stream through that fallback.
- **GPIO chip** — on Pi 5 the RP1 controller is `/dev/gpiochip0`, but the lgpio pin factory also probes `gpiochip4`; map the host controller to **both** container nodes (shown below) or motors fail with `can not open gpiochip`. Run `gpiodetect` on the host to confirm which chip is `pinctrl-rp1`.

```bash
# Build (cross-compile on Mac/Linux, or build directly on the Pi)
docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .

# Recommended — docker compose wires up camera + both gpiochips + udev for you.
# --build is required the first time and after every git pull (stale-image trap).
SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up --build

# Equivalent single docker run (Pi 5) — replace 192.168.1.42 with your PC's IP.
# BOTH gpiochip mappings are required (RP1 is gpiochip0 but lgpio also probes 4),
# and -v /run/udev is required for libcamera to find the CSI camera.
docker run --rm --privileged --device /dev/video0:/dev/video0 --device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4 -v /run/udev:/run/udev:ro -e SERVER_IP=192.168.1.42 nav-robot
```

**Operator UI viewer — Streamlit (browser-based, recommended):**

The Streamlit viewer is bundled in the server Docker image and starts automatically via `start_server.sh`.  No installation needed on the client machine.

```
Open http://localhost:8501 in any browser after starting the container.
Enter the server IP (use "localhost" when the viewer runs on the same machine).
```

| Scenario | URL to open | Server IP to enter |
|---|---|---|
| Docker on Mac / Linux same machine | `http://localhost:8501` | `localhost` |
| Server in Docker Desktop on Mac | `http://localhost:8501` | `localhost` |
| Server on a remote LAN machine | `http://<server-LAN-IP>:8501` | `localhost` |

**Operator UI viewer — native desktop window (optional):**

Prefer an app-like window over a browser tab? Two native options — install the
viewer deps once with `pip install -r requirements_client.txt`:

```bash
cd Code/Client

# (a) Streamlit UI inside a native window (via pywebview) — same UI as the browser
python desktop_viewer.py
#   macOS / Windows: works out of the box.
#   Linux: also `sudo apt-get install -y python3-gi gir1.2-webkit2-4.1` once.

# (b) PyQt5 desktop viewer (adds keyboard shortcuts)
python ai_viewer.py
```

---

## Operator UI controls

The browser viewer, the native desktop window (both render the Streamlit UI), and
the PyQt5 desktop viewer provide the same controls.  Keyboard shortcuts are only
available in the PyQt5 viewer.

| Control | Streamlit | PyQt5 shortcut | Action |
|---|---|---|---|
| **AUTO MODE** | button | `Ctrl+A` | AI decision fuser drives the robot |
| **MANUAL MODE** | button | `Ctrl+M` | Operator controls motors directly; AI is paused |
| **PREDICTIVE** | button | `Ctrl+P` | Switch to V-JEPA 2 predictive mode |
| **BASELINE** | button | `Ctrl+B` | Switch to reactive-only baseline mode |
| **Drive buttons** | ▲▼◄► buttons | Arrow keys (hold) | Manual drive in MANUAL mode |
| **Speed: Full / Slow** | radio | — | Toggle between full PWM and slow PWM |
| **EMERGENCY STOP** | button | `Space` / `Esc` | Cut motor power and disable AI |
| **SHUTDOWN SERVER** | button | `Ctrl+Q` | Send `CMD_KILL` – stop motors, shut down server |

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
| `gpio.chip` | `0` | GPIO chip number — Pi 5 kernel ≥ 6.6 → `0`; Pi 5 kernel < 6.6 → `4`; Pi 4 → `0`. Run `gpiodetect` to confirm |
| `camera.stream_width/height` | `400 × 300` | Camera JPEG resolution |
| `robot.speed_full` | `1500` | Full-speed PWM |
| `robot.speed_slow` | `800` | Reduced-speed PWM |
| `ultrasonic.read_interval` | `0.1` | Seconds between CMD_SONIC sends |

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
- Robot TCP client protocol (including `send_sonic`)
- Robot connection server protocol (including CMD_SONIC parsing and `send_aimove`)

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
