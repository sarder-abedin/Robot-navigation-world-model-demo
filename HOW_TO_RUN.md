# How to Run — Tank Robot Predictive Navigation

> Part of the [Tank Robot](README.md) docs — see also [ARCHITECTURE.md](ARCHITECTURE.md) · [CALIBRATION.md](CALIBRATION.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

Three pieces on **one network**. Start them in this order: **server → robot → viewer**.

| Piece | Runs on | Does | Ports |
|---|---|---|---|
| **Server** | PC / laptop (GPU recommended) | **All** the AI — YOLO11n, V-JEPA 2, SSv2, depth, decisions | binds 5003/8003 (UI), 5004/8004 (robot), 8501 (browser UI) |
| **Robot** | Raspberry Pi | Thin client — streams camera + ultrasonic, drives motors. **No AI.** | none (connects *out* to the PC) |
| **Viewer** | any machine | Operator UI — video, status, maps, drive controls | connects to the PC |

No robot? Run the server in **demo mode** with a video file (see [§2](#2-server-pc)) — no Pi needed. Hit a camera/GPIO/Qt/OOM snag? → [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## 1. Prerequisites

Clone the repo on each machine that needs it (PC and Pi):

```bash
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo
```

Find the **PC's LAN IP** — the robot and a remote viewer connect to it (call it `<PC_IP>`):

```bash
hostname -I | awk '{print $1}'      # e.g. 192.168.68.114
```

All three devices must be able to reach each other; if the Pi/viewer can't connect, allow inbound `5003,8003,5004,8004,8501` on the PC's firewall.

---

## 2. Server (PC)

The server runs **either in Docker** (reproducible, recommended) **or in a native venv** (needed for a Mac GPU). Pick one, then run in **demo** (video file) or **live** (real robot) mode.

### 2.1 · Run with Docker

**Build** the image for your hardware (pick one):

```bash
# CPU-only (any Mac/Linux) — simplest:
docker build -f Dockerfile.server -t nav-server .

# AMD GPU (ROCm) — use the wheel index matching your ROCm (see §2.3):
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.3 \
             -f Dockerfile.server -t nav-server-rocm .

# NVIDIA GPU (CUDA) — run later with --gpus all:
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
             -f Dockerfile.server -t nav-server-gpu .
```

**Run — live mode** (waits for the Pi). This is the full AMD/ROCm command:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -e NAV_MODE=live \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -v hf-cache:/root/.cache/huggingface \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 \
  nav-server-rocm
```

- The AMD GPU flags are the first two lines (`--device=/dev/kfd --device=/dev/dri --group-add video` and `--security-opt seccomp=unconfined`). **CPU-only:** drop them and use image `nav-server`. **NVIDIA:** replace them with `--gpus all` and use image `nav-server-gpu`.
- `-e NAV_MODE=live` — **required for a real robot**; without it the image defaults to *demo* and dies with `Cannot open demo video`.
- `-e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — AMD only; keeps V-JEPA 2's attention on the GPU instead of OOMing (see §2.3). Harmless on CPU/NVIDIA.
- `-v hf-cache:…` — model weights (a few hundred MB) download **once** and persist across `--rm` runs.

**Run — demo mode** (no robot; needs a corridor video):

```bash
mkdir -p assets/demo_clips && cp /path/to/corridor.mp4 assets/demo_clips/   # any 30 s indoor clip
docker run --rm \
  -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 \
  -v "$(pwd)/assets:/app/assets:ro" -v "$(pwd)/logs_rpi:/app/logs_rpi" \
  nav-server
```

**Or via compose** (forwards `NAV_MODE`/`NAV_STRATEGY`; `--build` the first time and after every `git pull`):

```bash
NAV_MODE=live NAV_STRATEGY=predictive docker compose -f docker-compose.server.yml up --build
```

The container starts the AI server **and** the browser UI (port 8501) together.

### 2.2 · Run natively (Python venv)

Needed for a **Mac GPU (MPS)** — Docker on macOS has no Metal passthrough (see §2.3). Also handy for development.

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_server.txt

cd Code/Server
python main_server.py --mode live --nav predictive          # live (real robot)
python main_server.py --mode demo --nav predictive           # demo (video file)
#   flags: --nav baseline (reactive only) · --video <path> · --no-display (headless)
```

### 2.3 · GPU & device notes

The heavy models (V-JEPA 2, SSv2, depth) read `device:` from `Code/Server/config.yaml`, default **`auto`** = **CUDA/ROCm → MPS → CPU**. Force one with `cuda`/`mps`/`cpu`. The startup line `V-JEPA 2 loaded: … on <device>` confirms the pick.

- **Apple Silicon (MPS):** run **natively** (§2.2), not in Docker. `auto` picks `mps`; the server sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops Metal lacks run on CPU. Confirm: `… on mps`.
- **AMD (ROCm):** ROCm registers as CUDA, so `auto`/`cuda` uses the Radeon with no code change. Install ROCm for your card (RX 9060 XT / RDNA 4 needs **ROCm 7.2+**), add yourself to the GPU groups (`sudo usermod -aG render,video $USER`; reboot). For a native venv, install the ROCm torch first: `pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch torchvision`, then the rest.
  - **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is important.** PyTorch ships the memory-efficient SDPA kernels **disabled** on AMD, so V-JEPA 2's attention falls back to the math backend that materialises the full N×N matrix and **OOMs a 16 GB card**. The repo sets this env for you (Dockerfile `ENV` + `device_utils`); pass `-e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` explicitly on an older image. **If AOTriton isn't available for your card, the forward falls back to CPU (slower) — never a crash.** Confirm: `… on cuda (bfloat16)` and **no** `ran out of GPU memory`.
  - If `torch.cuda.is_available()` is `False`, set `HSA_OVERRIDE_GFX_VERSION=12.0.0` (RDNA 4 = gfx12); usually unnecessary on native ROCm 7.2.
- **NVIDIA (CUDA):** `auto` selects CUDA. In Docker, build the CUDA variant (§2.1) and run with `--gpus all` (needs the nvidia-container-toolkit).
- **World model runs in a separate process** (`world_model.run_in_subprocess: true`): V-JEPA 2 + SSv2 hold the GIL for seconds; a subprocess keeps the camera I/O smooth. You'll see `World-model subprocess started` then `… ready` (~a minute); until then `wm=` risk is 0 and the detector drives. Set `false` to run in-thread (simpler, can stutter the stream).

### 2.4 · Confirm the server is up

- Logs reach `V-JEPA 2 loaded: … on <device>` and then the server **waits for the Pi** (live) or starts processing the video (demo).
- Browser UI answers at `http://<PC_IP>:8501` (or `http://localhost:8501` on the PC itself).

---

## 3. Robot (Raspberry Pi)

The Pi runs **in Docker** as a thin client (picamera2 + libcamera + GPIO; **no** torch/ultralytics). It connects *out* to the PC and binds no ports.

### 3.1 · Build the image (on the Pi)

```bash
git pull                                          # get the latest fixes first
docker build -f Dockerfile.robot -t nav-robot .   # native arm64 build on the Pi
```
Cross-building on an amd64 PC instead? Add `--platform linux/arm64` (needs `buildx` + QEMU; slow because the camera wheels compile — building **on the Pi** is recommended).

### 3.2 · Run it

Only **one** process may hold the camera + GPIO at a time. First, sanity-check the camera on the Pi host, and make sure no old robot container is running:

```bash
rpicam-hello --list-cameras                                  # should list your camera (e.g. ov5647)
docker rm -f $(docker ps -q --filter ancestor=nav-robot) 2>/dev/null   # stop any previous one
```

Then start it with compose (it mounts the camera, media nodes, GPIO, udev **and** `/dev/dma_heap` for you). Use your real IP and `--build` after a `git pull`:

```bash
SERVER_IP=<PC_IP> docker compose -f docker-compose.robot.yml up --build
```

<details><summary>Equivalent single <code>docker run</code></summary>

```bash
docker run --rm --privileged \
  --device /dev/video0:/dev/video0 \
  --device /dev/media0:/dev/media0 --device /dev/media1:/dev/media1 --device /dev/media2:/dev/media2 \
  --device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4 \
  -v /run/udev:/run/udev:ro \
  -v /dev/dma_heap:/dev/dma_heap \
  -e SERVER_IP=<PC_IP> \
  nav-robot
```

Why each mount: `-v /run/udev` + the `/dev/media*` nodes let libcamera find the CSI camera; **both** gpiochip maps are required (RP1 is `gpiochip0`, lgpio also probes `gpiochip4`; run `gpiodetect` to confirm the `pinctrl-rp1` chip); `-v /dev/dma_heap` gives picamera2 the cached buffers — without it the stream can **stall/freeze** after a while.
</details>

### 3.3 · Confirm the robot is streaming

- **Success:** no `Device or resource busy` / `GPIO busy`, and the **server** log starts printing `SCENE:` / detection lines.
- **`busy`?** Another robot process owns the hardware — stop duplicate containers (command above), and if PipeWire is holding the camera see [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Reboot the Pi if unsure, then run **one** container.
- **`camera frame is stale` on the PC after a while?** You're on an old image — `git pull` + rebuild (`--build`) so you have the DMA-heap mount + stall auto-restart.

<details><summary>Bare-metal Pi install (unsupported fallback, no Docker)</summary>

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-libcamera python3-gpiozero python3-kms++ python3-prctl libatlas-base-dev
python3 -m venv .venv --system-site-packages   # picamera2/libcamera come from apt
source .venv/bin/activate
pip3 install -r Code/Robot/requirements_robot.txt
python3 Code/Robot/main_robot.py --server-ip <PC_IP>
```
</details>

---

## 4. Viewer

Two choices. The **PyQt viewer** is the only one with the live **world map + V-JEPA 2 foresight**; the **browser** UI is zero-install and display-only. Both show the video, status, and drive controls.

### 4.1 · Browser (Streamlit) — zero install

Bundled in the server image and started automatically. Just open:

```
http://<PC_IP>:8501          (or http://localhost:8501 on the server PC)
```

### 4.2 · PyQt desktop viewer (`ai_viewer.py`) — full maps

```bash
cd Code/Client
pip install -r ../../requirements_client.txt      # opencv-python-headless, PyQt5, numpy, …
python3 ai_viewer.py
```
Then set **PC Server IP** (`127.0.0.1` if on the server PC, else `<PC_IP>`) and click **Connect**.

- **Linux/Wayland**, if it aborts with `Could not load the Qt platform plugin "xcb"`, run on Wayland instead:
  ```bash
  QT_QPA_PLATFORM=wayland python3 ai_viewer.py
  # permanent:  echo 'export QT_QPA_PLATFORM=wayland' >> ~/.bashrc
  ```
  (xcb route + the exact missing X libs: see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).)

### 4.3 · Native desktop window (optional)

Same UI as the browser, in an OS window via pywebview:

```bash
cd Code/Client && python desktop_viewer.py
#   macOS/Windows: works out of the box.
#   Linux venv backend:  pip install qtpy PyQtWebEngine   (or apt: python3-gi gir1.2-webkit2-4.1)
```

### 4.4 · Which server IP to enter

| Viewer runs… | Enter |
|---|---|
| On the **server PC** | `127.0.0.1` / `localhost` |
| On **another LAN machine** | the PC's LAN IP (`<PC_IP>`) |
| Server in **Docker Desktop** on a Mac | `localhost` |

---

## 5. Drive

In the viewer. The **PREDICTIVE / BASELINE** buttons stay greyed until you pick a Navigation Mode — that's the safety gate.

| To do this | Click, in order |
|---|---|
| **Obstacle Avoidance** (wander, avoid obstacles) | *Obstacle Avoidance* radio → **PREDICTIVE** |
| **Goal Following** (drive to a point) | *Goal Following* radio → **Set Goal** → click a point on the video → **PREDICTIVE** |
| **Manual** (you drive) | **MANUAL** → arrow keys / on-screen D-pad (Full/Slow speed) |
| **Emergency stop** (anytime) | **Space** or **Esc** |

Keyboard shortcuts (PyQt viewer only): `Ctrl+A` AUTO · `Ctrl+M` MANUAL · `Ctrl+P` predictive · `Ctrl+B` baseline · `Space`/`Esc` emergency stop · `Ctrl+Q` shut down server.

Below the video are two maps: the **local** map (instant depth/sonar/goal, robot-centred) and the **world** map (accumulating trajectory + ultrasonic + YOLO objects + V-JEPA 2 foresight diamonds; **Reset trail** clears it). PREDICTIVE = full stack incl. V-JEPA 2; BASELINE = reactive only.

> The video is black until the robot streams frames (§3). No frames → the AI has nothing to act on and won't drive — so if it "does nothing", check the robot.

---

## 6. Calibration

Some signals are only *qualitatively* right until calibrated for your robot + space — the V-JEPA 2 risk/label, the governor's metres, and the depth scale. The reroute direction and the ultrasonic hard-stop work **without** calibration. Full guide: **[CALIBRATION.md](CALIBRATION.md)**.

**Fastest path — zero extra driving:** enable `logging.save_raw_frames`, do a normal run (logging on, working sonar), then calibrate at your desk with the guided PyQt UI (never drives the robot):

```bash
python Code/Server/calibration_ui.py
```
Numbered steps with MANDATORY/RECOMMENDED/OPTIONAL badges: **1** select run(s) → **2** analyze → **3** depth scale, **4** governor speeds, **5** anchors → **6** apply to config → **7** restart the server. CLI equivalent: `python Code/Server/calibrate_from_logs.py --run ../../logs_rpi/<run> --anchors --apply config.yaml`.

**Manual, in-corridor** (do them in this order):

```bash
# 1. V-JEPA 2 anchors (biggest quality win) — where V-JEPA 2 loads (GPU box)
python Code/Server/calibrate_anchors.py --blocked ./blocked --clear ./clear --out anchors.npz
#    → config.yaml: world_model.anchors_path: "anchors.npz"
# 2. Depth scale — measure a known distance, read the HUD → depth.scale: actual/reported
# 3. Governor speeds — on the robot, facing a flat wall with ~2–3 m runway:
python Code/Robot/calibrate_governor.py --apply ../Server/config.yaml
```

---

## 7. Configuration

### PC server (`Code/Server/config.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `navigation_mode` | `predictive` | Starting navigation mode |
| `world_model.device` / `depth.device` / `ssv2.device` | `auto` | Compute device: CUDA/ROCm → MPS → CPU (or force `cuda`/`mps`/`cpu`) |
| `world_model.precision` / `depth.precision` / `ssv2.precision` | `bf16` | GPU compute dtype (bf16/fp16/fp32; CPU/MPS → fp32) |
| `world_model.gpu_retry_every_calls` | `30` | After a GPU OOM, forwards run on CPU; re-probe the GPU every N calls (0 = stay on CPU) |
| `world_model.risk_similarity_threshold` | `0.55` | V-JEPA 2 BLOCKED sensitivity |
| `world_model.label_margin` | `0.02` | Obstacle-vs-clear gap for a BLOCKED/CLEAR label (below → MIXED). Calibrate anchors |
| `world_model.anchors_path` | `""` | Calibrated corridor anchors (`calibrate_anchors.py`); empty = synthetic |
| `decision.weights.world_model` | `0.45` | V-JEPA 2 contribution to fused risk |
| `decision.low_risk_max` | `0.25` | Below this → FORWARD |
| `decision.medium_risk_max` | `0.50` | Below this → SLOW, above → active avoidance |
| `decision.reroute.closed_loop` | `true` | Dynamic wait/turn-until-clear/backup avoidance (vs legacy one-shot reroute) |
| `decision.reroute.wait_timeout_seconds` | `2.0` | WAIT for a crossing obstacle at most this long, then TURN |
| `decision.reroute.max_turn_seconds` | `4.0` | Keep turning at most this long, then STOP & reassess |
| `decision.reroute.backup_distance_m` | `0.35` | Closer than this + approaching → BACKUP |
| `decision.reroute.direction_margin_m` / `_frac` | `0.05` / `0.10` | A side must beat centre free-space by this absolute distance **or** relative fraction to TURN toward it |
| `decision.reroute.ultrasonic_escalate_seconds` | `1.5` | Hold the ultrasonic reflex STOP this long; then escalate to a maneuver (a sonar wall never raises the vision risk) |
| `decision.reroute.ultrasonic_resume_risk` | `0.5` | Resume FORWARD only when ultrasonic risk drops below this (hysteresis) |
| `decision.reroute.dynamic_classes` | `[person, cat, dog]` | Obstacles likely to move (→ WAIT) |
| `temporal_action.depth_presence_range_m` | `1.5` | Depth obstacle within this range feeds the motion recogniser when YOLO is blind |
| `decision.governor.enabled` | `true` | Kinematic latency-aware safe-speed governor |
| `decision.governor.forward_speed_mps` / `slow_speed_mps` / `max_decel_mps2` | `0.35` / `0.18` / `0.6` | **Calibrate** — measured speeds + hardest deceleration |
| `pose.turn_rate_dps` / `pose.backup_speed_mps` | `45.0` / `0.04` | **Calibrate** — turn rate + reverse speed for the dead-reckoned world map |
| `robot.ultrasonic_stop_cm` | `30.0` | Ultrasonic hard-stop distance (cm) — deterministic reflex |
| `camera.clip_length` / `world_model.input_size` | `64` / `256` | Match the V-JEPA 2 checkpoint (`vitl-fpc64-256`) |
| `depth.enabled` / `depth.model_id` | `true` / `…Metric-Indoor-Small-hf` | Depth-Anything free-space channel |
| `depth.run_every_n_frames` / `world_model.run_every_n_frames` | `6` / `8` | Inference cadence (auto-raised on CPU) |
| `server.cmd_port` / `video_port` / `robot_cmd_port` / `robot_video_port` | `5003` / `8003` / `5004` / `8004` | TCP ports |

### Pi robot (`Code/Robot/config_robot.yaml`)

| Setting | Default | Effect |
|---|---|---|
| `server_ip` | `192.168.1.100` | PC server IP (overridable with `-e SERVER_IP` / `--server-ip`) |
| `gpio.chip` | `0` | GPIO chip — Pi 5 kernel ≥ 6.6 → `0`; < 6.6 → `4`; Pi 4 → `0`. Run `gpiodetect` |
| `camera.stream_width/height` | `400 × 300` | Camera JPEG resolution |
| `camera.stall_timeout_seconds` / `watchdog_interval_seconds` | `2.5` / `1.0` | Auto-restart the camera if the encoder stalls |
| `robot.speed_full` / `speed_slow` | `1500` / `800` | Motor PWM duty |
| `ultrasonic.read_interval` | `0.1` | Seconds between `CMD_SONIC` sends |
| `network.io_timeout_seconds` / `video_timeout_seconds` | `4.0` / `20.0` | Command vs video socket timeouts (auto-reconnect) |

---

## 8. Tests

No GPU or hardware needed:

```bash
pytest tests_rpi/ -v      # decision fusion, maps, pose, camera-stall, device utils, protocol …
pytest tests/ -v          # original Freenove tests
```
Pre-existing failures in `test_world_model_rpi.py` are expected when `torch` isn't installed.

---

## 9. Logging

Each run writes a timestamped directory under `logs_rpi/`:

```
logs_rpi/run_YYYYMMDD_HHMMSS_predictive/
├── navigation_log.csv   # one row per frame (detector/world_model/temporal risk kept separate)
├── system.log
├── frames/              # annotated JPEGs (every 5th frame)
├── raw_frames/          # raw JPEGs — only if logging.save_raw_frames (for anchors)
└── viz/                 # PNG plots — from the run visualizer's "Save"
```

Logging is server-side; toggle it live from the UI (on by default) or with `--logging on|off` / `NAV_LOGGING=1`.

**Visualise a run (offline):**

```bash
python Code/Server/run_visualizer.py     # pick a run → synced plots + frame scrubber → Save PNGs
```
