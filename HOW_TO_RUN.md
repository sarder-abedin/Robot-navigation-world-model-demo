# How to Run — Tank Robot Predictive Navigation

> Part of the [Tank Robot](README.md) docs — see also [ARCHITECTURE.md](ARCHITECTURE.md) · [HOW_TO_RUN.md](HOW_TO_RUN.md) · [CALIBRATION.md](CALIBRATION.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Everything operational: setup, the three run modes (demo / live / Docker), the UI
viewer options, anchor + governor calibration, configuration tables, tests and
logging. Hit a camera/GPIO/compose snag? See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Quick start — full system (AMD GPU PC + Pi robot + viewer)

The copy-paste happy path for a real robot with an **AMD/ROCm** server. Three
devices on the **same network**: the **PC** (AMD GPU) runs *all* the AI, the
**Raspberry Pi** is a thin client that connects *out* to the PC, and the **viewer**
is the operator UI. Do the steps **in this order**. (Demo mode, NVIDIA, and the
Python-venv path are further down.)

### 0 · Find the PC's LAN IP (on the PC)

```bash
hostname -I | awk '{print $1}'      # e.g. 192.168.68.114 — call this <PC_IP>
```

### 1 · Start the server (PC, AMD GPU / ROCm)

Build once, then run in **live** mode. ⚠️ The image defaults to *demo* mode, which
needs a bundled video and dies with `Cannot open demo video` — for a real robot you
**must** pass `-e NAV_MODE=live`:

```bash
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.3 \
             -f Dockerfile.server -t nav-server-rocm .

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -e NAV_MODE=live \
  -v hf-cache:/root/.cache/huggingface \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 \
  nav-server-rocm
```

- `-e NAV_MODE=live` — wait for the robot instead of opening a demo video.
- `-v hf-cache:…` — the model weights (V-JEPA 2 / Depth / SSv2, a few hundred MB)
  download **once** and persist, instead of re-downloading every `--rm` run.
- **Confirm the GPU:** the logs show `V-JEPA 2 loaded: … on cuda (bfloat16)`. `cuda`
  is your Radeon (ROCm reports as CUDA); `bfloat16` means the memory-safe path is on.
  In another terminal, `rocm-smi` shows VRAM/utilisation rising during inference.
- First run downloads the models (a few minutes); then the server **waits for the Pi**.

### 2 · Start the robot (Raspberry Pi)

Only **one** process may hold the camera + GPIO at a time. Sanity-check the camera on
the Pi host first:

```bash
rpicam-hello --list-cameras     # should list your camera (e.g. ov5647)
```

Then start the client — compose mounts the camera, media nodes, GPIO **and** udev:

```bash
SERVER_IP=<PC_IP> docker compose -f docker-compose.robot.yml up
```

<details><summary>Equivalent single <code>docker run</code></summary>

```bash
docker run --rm --privileged \
  --device /dev/video0:/dev/video0 \
  --device /dev/media0:/dev/media0 --device /dev/media1:/dev/media1 \
  --device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4 \
  -v /run/udev:/run/udev:ro \
  -e SERVER_IP=<PC_IP> \
  nav-robot
```

`-v /run/udev` **and** the `/dev/media*` devices are what let libcamera find the CSI
camera; both gpiochip maps are required (RP1 is gpiochip0, lgpio also probes 4).
</details>

- **Success:** no `Device or resource busy` / `GPIO busy`, and the **server** log
  starts printing detections/scene lines.
- **`busy`?** Another robot container is still running. Stop it and retry:
  ```bash
  docker rm -f $(docker ps -q --filter ancestor=nav-robot)   # then re-run step 2
  ```

### 3 · Start the viewer

Pick **one**. The **PyQt viewer** is the one with the live **world map + V-JEPA 2
foresight**; the browser UI is display-only.

**A · Browser (no install):** open `http://<PC_IP>:8501`.

**B · PyQt viewer (`ai_viewer.py`) — full maps:**

```bash
cd Code/Client
pip install -r ../../requirements_client.txt      # opencv-python-headless, PyQt5, …
python3 ai_viewer.py
```

- **Linux/Wayland:** if it aborts with `Could not load the Qt platform plugin "xcb"`,
  run it on Wayland instead (fastest fix):
  ```bash
  QT_QPA_PLATFORM=wayland python3 ai_viewer.py
  # make it permanent:  echo 'export QT_QPA_PLATFORM=wayland' >> ~/.bashrc
  ```
  (To use xcb instead, install the X libs — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).)
- In the window, set **PC Server IP** — `127.0.0.1` if the viewer runs **on** the
  server PC, otherwise `<PC_IP>` — and click **Connect**.

### 4 · Drive (in the viewer)

The **PREDICTIVE / BASELINE** buttons stay greyed until you pick a Navigation Mode —
that's the safety gate, not a bug.

| To do this | Click, in order |
|---|---|
| **Obstacle Avoidance** (wander, avoid obstacles) | *Obstacle Avoidance* radio → **PREDICTIVE** |
| **Goal Following** (drive to a point) | *Goal Following* radio → **Set Goal** → click a point on the video → **PREDICTIVE** |
| **Manual** (you drive) | **MANUAL** → arrow keys / on-screen D-pad (Full/Slow speed) |
| **Emergency stop** (anytime) | **Space** or **Esc** |

Watch the two maps below the video: the **local** map (instant depth/sonar/goal) and
the **world** map (accumulating trajectory + ultrasonic + YOLO objects + V-JEPA 2
foresight diamonds; **Reset trail** clears it).

> The video is black until the robot streams frames (step 2). With no frames the AI
> has nothing to act on and won't drive — so if it "does nothing", check the robot.

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

#### Compute device (`device: auto`)

The heavy models (V-JEPA 2 + SSv2) and the depth channel default to `device: "auto"`
in `Code/Server/config.yaml`, which resolves **CUDA/ROCm → MPS → CPU** and uses the
best available accelerator — no per-machine edit needed. Force one with
`"cuda"` / `"mps"` / `"cpu"` (each degrades gracefully if unavailable). On startup the
log line `V-JEPA 2 loaded: … on <device>` confirms what it picked.

#### Apple Silicon GPU (MPS)

On an M-series Mac, `auto` selects the **Apple GPU (Metal / MPS)** — lower latency,
fewer dropped camera frames, faster STOP, and room to run V-JEPA 2 more often for
earlier predictive warnings.

- **Run the server NATIVELY (Option 1 venv), not in Docker.** Docker on a Mac has no
  Metal passthrough, so a container always falls back to CPU regardless of `device`.
- The native `pip install -r requirements_server.txt` gives you an MPS-capable
  `torch` (any `torch>=2.1`); nothing extra to install.
- The server enables `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically when it selects
  MPS, so the handful of ops Metal doesn't implement run on CPU instead of crashing.
- Confirmation line: `V-JEPA 2 loaded: … on mps`.

#### AMD GPU (ROCm)

ROCm's PyTorch registers the AMD GPU **through the CUDA API**, so `device: "auto"`
(or `"cuda"`) uses it with **no code changes** — `torch.cuda.is_available()` is `True`
and the models run on the Radeon. Steps (native Linux is the reliable path — not
macOS/Windows):

1. **Install ROCm** for your card on a supported Linux (Ubuntu 22.04/24.04, etc.). The
   RX 9060 XT / RDNA 4 needs **ROCm 7.2+**. Add yourself to the GPU groups and reboot:
   ```bash
   sudo usermod -aG render,video $USER
   ```
2. **Install the ROCm build of PyTorch** (it bundles the ROCm runtime), then the rest
   of the deps *without* re-installing torch (the `torch<2.8` pin in
   `requirements_server.txt` only exists to keep the Mac Docker image CPU-only):
   ```bash
   python3 -m venv roboenv && source roboenv/bin/activate
   pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch torchvision
   pip install transformers accelerate ultralytics opencv-python-headless \
               numpy pyyaml pillow streamlit matplotlib PyQt5
   ```
   (Use the wheel index matching your installed ROCm; a very new RDNA-4 card may need
   the nightly index.)
3. **Verify + run:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   cd Code/Server && python main_server.py --mode live --nav predictive
   # startup logs: V-JEPA 2 loaded: … on cuda   ← the Radeon via ROCm
   ```
4. **Docker** (optional): build with the ROCm wheel index and pass the AMD devices —
   ```bash
   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.3 \
                -f Dockerfile.server -t nav-server-rocm .
   # Add -e NAV_MODE=live for a real robot (default is demo, which needs a video),
   # and -v hf-cache:… so the weights download only once. See the Quick start above.
   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
              --security-opt seccomp=unconfined -e NAV_MODE=live \
              -v hf-cache:/root/.cache/huggingface \
              -p 5003:5003 -p 8003:8003 \
              -p 5004:5004 -p 8004:8004 -p 8501:8501 nav-server-rocm
   ```
   (If the base image lacks ROCm userspace, base it on `rocm/pytorch` instead.)

**Gotchas:** if `torch.cuda.is_available()` is `False`, the card's gfx target may not
be auto-detected — set `HSA_OVERRIDE_GFX_VERSION=12.0.0` (RDNA 4 = gfx12); with native
ROCm 7.2 support this is usually unnecessary. The world-model subprocess uses `spawn`,
which is ROCm/CUDA-safe. ROCm is typically **more stable than MPS** (no Metal segfaults)
and decouples the AI from the Mac.

#### NVIDIA GPU (CUDA)

On an NVIDIA host, `auto` selects **CUDA**. For Docker, build the CUDA torch variant
with the `TORCH_INDEX` build-arg and run with `--gpus all` (see
[Option C – Docker](#option-c--docker-recommended-for-reproducibility)).

#### World model runs in a separate process

V-JEPA 2 (ViT-L over a 64-frame clip) and SSv2 hold the Python GIL for seconds per
call, which — on a background thread — froze the whole server and stalled the
robot's camera stream until it timed out. They now run in a **separate OS process**
(`Code/Server/world_model_process.py`, `world_model.run_in_subprocess: true`), so the
main loop only pays a tiny per-clip cost and the camera I/O stays smooth.

- On startup you'll see `World-model subprocess started (pid=…)` and, once its models
  finish loading + warming (~a minute), `World-model subprocess ready`.
- Until then the world-model risk `wm=` stays 0 and the drive loop uses the detector's
  instantaneous risk — the robot still drives; V-JEPA 2 just contributes a bit later.
- Depth stays inline (it's lighter and decision-critical). If the subprocess dies, the
  world model degrades to off (detector risk) with a logged hint — nav still runs.
- Set `world_model.run_in_subprocess: false` to run it on an in-process thread instead
  (simpler, but the camera stream can stutter during inference).

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

In demo mode the server runs local YOLO11n (no Pi needed).

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
> AUTO/MANUAL controls. The **PyQt5 `ai_viewer.py`** also draws a live **2D
> navigation map** beside the video — a top-down, robot-centred view of the depth
> free-space (left/centre/right), the ultrasonic reading, the chosen clear
> direction and the goal (by bearing + distance). It's *egocentric* (there's no
> odometry), so it shows the robot's local surroundings right now, not a
> world-fixed map; toggle it with "Show 2D navigation map". Inside Docker the
> **browser** viewer starts automatically (a native window needs a display, so
> `desktop_viewer.py` / `ai_viewer.py` are native-only).

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
1. Streams JPEG frames to the PC (the PC runs YOLO11n + V-JEPA 2 on them)
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

## Calibration

Several signals are only *qualitatively* right until calibrated for your robot and
space — the V-JEPA 2 risk/label, the speed governor's metres, and the depth scale.
The full step-by-step (prerequisites, verification, troubleshooting) lives in
**[CALIBRATION.md](CALIBRATION.md)**.

**Fastest path — zero extra driving:** turn on `logging.save_raw_frames`, do one
or more normal runs (logging on, working sonar), then calibrate at your desk with
the **separate calibration UI** (a PyQt5 window that never drives the robot):
```bash
python Code/Server/calibration_ui.py
```
It's a **step-by-step guided workflow** — each numbered step carries a
**MANDATORY / RECOMMENDED / OPTIONAL** badge and a live status (done / do this next
/ waiting / not-ready) so you always know the next move (soft guidance — it never
blocks you from going out of order): **1 · Select run(s)** (tick one; pooling more
is optional) → **2 · Analyze** → **3 · Depth scale**, **4 · Governor speeds**,
**5 · Anchors** (readouts, all recommended) → **6 · Apply to config** (verifies) →
**7 · Restart the server**.
Or the CLI: `python Code/Server/calibrate_from_logs.py --run ../../logs_rpi/<run> [more…] --anchors --apply config.yaml`.
See CALIBRATION.md → "Zero-driving calibration from logs".

The manual, in-corridor essentials:

```bash
# 1. V-JEPA 2 anchors (biggest quality win) — run where V-JEPA 2 loads (GPU/MPS box)
cd Code/Server
python calibrate_anchors.py --blocked ./blocked --clear ./clear --out anchors.npz
#   → config.yaml: world_model.anchors_path: "anchors.npz"

# 2. Depth scale — measure a known distance, read the HUD, set scale = actual/reported
#   → config.yaml: depth.scale: <actual/reported>

# 3. Speed governor — ON THE ROBOT, facing a flat wall with ~2–3 m runway
cd Code/Robot
python calibrate_governor.py --apply ../Server/config.yaml   # measure + patch safely
```

Do them in that order. The **reroute direction** (relative depth) and the
**ultrasonic hard-stop** (sonar) work *without* calibration — only the absolute-
distance signals (world-model risk, governor metres, goal-arrival distance) need it.
See **[CALIBRATION.md](CALIBRATION.md)** for details.

---

## Configuration

### PC server (`Code/Server/config.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `navigation_mode` | `predictive` | Starting navigation mode |
| `world_model.risk_similarity_threshold` | `0.55` | V-JEPA 2 BLOCKED sensitivity |
| `world_model.label_margin` | `0.02` | Obstacle-vs-clear similarity gap for a BLOCKED/CLEAR label (relative test; below → MIXED). Calibrate anchors for meaningful labels |
| `decision.weights.world_model` | `0.45` | V-JEPA 2 contribution to fused risk |
| `decision.low_risk_max` | `0.25` | Below this → FORWARD |
| `decision.medium_risk_max` | `0.50` | Below this → SLOW, above → active avoidance |
| `decision.reroute.closed_loop` | `true` | Dynamic wait/turn-until-clear/backup avoidance (vs legacy one-shot reroute) |
| `decision.reroute.wait_timeout_seconds` | `2.0` | WAIT for a crossing obstacle at most this long, then TURN |
| `decision.reroute.max_turn_seconds` | `4.0` | Keep turning at most this long, then STOP & reassess |
| `decision.reroute.backup_distance_m` | `0.35` | Closer than this + approaching → BACKUP |
| `decision.reroute.direction_margin_m` | `0.05` | A side must beat centre free-space by this **absolute** distance to TURN toward it (else STOP/search) |
| `decision.reroute.direction_margin_frac` | `0.10` | …OR by this **relative** fraction of the centre distance. The relative test makes reroute work on **uncalibrated** depth (per-side gaps of a few cm/percent); the old absolute-only 0.3 m was unreachable, so the robot never turned and sat in STOP |
| `decision.reroute.ultrasonic_escalate_seconds` | `1.5` | Hold the ultrasonic reflex STOP this long; if the obstacle won't clear, escalate to a maneuver (turn/back-up/search) — a wall the sonar sees never raises the *vision* risk, so without this the robot just sits stopped |
| `decision.reroute.ultrasonic_resume_risk` | `0.5` | Once maneuvering around a sonar obstacle, resume FORWARD only when ultrasonic risk drops below this (front clear by a margin). Hysteresis that stops the forward/backward oscillation at the stop threshold; lower = require more clearance |
| `temporal_action.depth_presence_range_m` | `1.5` | Depth obstacle within this range (m) feeds the motion recogniser when YOLO is blind — but only if the centre is clearly nearer than the sides (a real obstacle, not a uniformly-close/mis-scaled reading) |
| `decision.reroute.dynamic_classes` | `[person, cat, dog]` | Obstacles likely to move out of the way (→ WAIT) |
| `decision.governor.enabled` | `true` | Kinematic safe-speed governor (proactive, latency-aware) |
| `decision.governor.forward_speed_mps` | `0.35` | **Calibrate** — robot speed at FORWARD (m/s) |
| `decision.governor.slow_speed_mps` | `0.18` | **Calibrate** — robot speed at SLOW (m/s) |
| `decision.governor.max_decel_mps2` | `0.6` | **Calibrate** — hardest deceleration (m/s²) |
| `decision.governor.target_speed_mps` | `0.0` | Speed before contact (0 = stop; >0 = reduce impact) |
| `robot.ultrasonic_stop_cm` | `30.0` | Ultrasonic hard-stop distance (cm) — deterministic reflex |
| `camera.clip_length` / `camera.ai_frame_size` | `64` / `256` | Match the V-JEPA 2 checkpoint (`vitl-fpc64-256`) |
| `world_model.anchors_path` | `""` | Calibrated corridor anchors (`calibrate_anchors.py`); empty = synthetic |
| `depth.enabled` | `true` | Depth-Anything free-space channel (metric distance + clear direction) |
| `depth.model_id` | `…Metric-Indoor-Small-hf` | Monocular depth checkpoint (metric indoor) |
| `depth.run_every_n_frames` | `6` | Depth cadence (auto-raised on CPU) |
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
    ├── frames/              # annotated JPEGs (every 5th frame)
    ├── raw_frames/          # raw JPEGs — only if logging.save_raw_frames (for anchors)
    └── viz/                 # PNG plots — written by the run visualizer's "Save"
```

The CSV captures all three risk signals separately (`detector_risk`,
`world_model_risk`, `temporal_risk`) so you can plot predictive vs baseline risk
trajectories from the same scenario.

### Visualize a run (offline)

```bash
python Code/Server/run_visualizer.py
```
A desk-only PyQt5 window: pick a run → see the risk / distance / action-timeline /
latency / network plots (zoom/pan) with a **synced annotated-frame scrubber** (drag
to move a time cursor across all charts) → **Save PNGs** writes one image per chart
plus `summary.txt` into `<run>/viz/`. Headless alternative:
`python -c "import sys; sys.path.insert(0,'Code/Server'); import run_report; run_report.save_pngs('logs_rpi/<run>')"`.
