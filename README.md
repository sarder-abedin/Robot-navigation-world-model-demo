# Freenove Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot Kit for
Raspberry Pi (FNK0077)** that uses **V-JEPA 2** as a world model to anticipate
future obstacles — not just react to what is currently visible.

---

## What each AI model does

### YOLOv8n — "The Photographer" (runs on Raspberry Pi 5)

Looks at one camera frame and draws boxes around anything in the path.

- **Anchor-free**: predicts object centres directly, no predefined box templates
- Runs every 2 frames to save Pi CPU
- Output: bounding boxes, labels, confidence scores, raw risk score 0–1
- **Limitation**: only sees what is already in front of the camera. A person
  entering at the edge produces a small, off-centre box → low risk score.
  By the time YOLO raises the alarm the obstacle is already blocking the path.

### V-JEPA 2 — "The Fortune Teller" (runs on operator laptop / PC)

Watches the last 16 frames and predicts what the scene will **feel like**
half a second from now — without drawing pixels.

- Compresses each frame into a high-dimensional "scene vibe" vector
- Masks the last 4 frames and asks: *"given frames 1–12, what should
  frames 13–16 look like?"*
- Compares the predicted future embedding to two reference prototypes:
  - `obstacle_anchor` — average embedding of corridor-blocked scenes
  - `clear_anchor` — average embedding of empty-corridor scenes
- Cosine similarity difference → `predicted_risk ∈ [0, 1]`

> **Note:** "anchor" here means *reference prototype in embedding space*,
> completely unrelated to YOLOv8's anchor-free detection head. Two
> different uses of the same word.

**Why this matters:** a person entering at the frame edge produces a tiny YOLO
box (low risk) but shifts the predicted future embedding toward the blocked
prototype. V-JEPA 2 raises the alarm several frames before YOLO does.

### SSv2 Temporal Recogniser — "The Behaviour Analyst" (runs on laptop / PC)

Watches the history of YOLO detections over 10 frames and classifies what
the obstacle is *doing* — no neural network, pure maths (linear regression
on bbox size and position).

| Pattern | What it sees | Risk |
|---|---|---|
| `STATIC_CLEAR` | No detection for most frames | 0.00 |
| `CLEARING` | Was there, now gone | 0.10 |
| `UNCERTAIN` | Not enough data | 0.25 |
| `CROSSING` | Moving sideways, stable size | 0.45 |
| `APPROACHING` | Growing area + centred | 0.70 |
| `BLOCKING` | Large, centred, stationary | 0.85 |

Inspired by the Something-Something V2 dataset philosophy: classify
*motion*, not objects.

### Decision Fuser — "The Judge" (runs on laptop / PC)

Combines all three risk signals into one action:

```
YOLO risk    × 0.35
V-JEPA 2     × 0.45   ← zeroed in baseline mode
Temporal     × 0.20
─────────────────────
Fused risk   → FORWARD (<0.30) / SLOW (<0.60) / STOP / REROUTE
```

A hysteresis filter (0.05 margin) prevents oscillation near thresholds.
V-JEPA 2 early-warning: if world model says `BLOCKED` but fused risk is
still in the FORWARD zone → proactively downgrade to SLOW.

**Baseline mode** zeroes V-JEPA 2's weight and halves temporal weight so
the comparison isolates exactly the V-JEPA 2 contribution.

---

## Project overview

The robot drives through a small indoor corridor (chairs, boxes, tape lines,
temporary occlusions, people crossing). Three AI layers run in parallel:

| Layer | Model | Role | Where it runs |
|---|---|---|---|
| Instantaneous detection | **YOLOv8n** | What is blocking the path *right now*? | **Server (Pi 5)** |
| Predictive world model | **V-JEPA 2** | What will the scene look like in ~0.5 s? | **Client (laptop/PC)** |
| Temporal motion pattern | **SSv2-style rules** | Is the obstacle approaching, crossing, or clearing? | **Client (laptop/PC)** |
| Decision fusion | Weighted risk fuser | Combine all three signals → action | **Client (laptop/PC)** |
| Motor execution | **Freenove tankMotor** | `setMotorModel(L, R)` | **Server (Pi 5)** |

The Pi handles fast reactive tasks (YOLOv8, ultrasonic, motors). The client
PC handles the heavy transformer inference (V-JEPA 2) where GPU headroom is
available and sends navigation commands back to the Pi.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Raspberry Pi 5  (Server)                                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Freenove existing stack (UNCHANGED)                        │    │
│  │  camera.py · car.py · motor.py · ultrasonic.py             │    │
│  │  server.py · tcp_server.py · command.py · message.py       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │ wraps / extends                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Pi-side AI (detection + hardware only)                     │    │
│  │                                                             │    │
│  │  main_predictive.py  ← entry point + TCP command handler   │    │
│  │  ai_pipeline.py      ← YOLOv8 loop + CMD_DETECTION bcast  │    │
│  │  camera_buffer.py    ← rolling JPEG→numpy frame buffer     │    │
│  │  detector.py         ← YOLOv8 obstacle detection           │    │
│  │  robot_control.py    ← wraps car.motor.setMotorModel()     │    │
│  │  ai_logger.py        ← CSV + annotated JPEG archive        │    │
│  │  visualization.py    ← OpenCV HUD (YOLO boxes + sonic)     │    │
│  │  config.yaml         ← Pi-side thresholds and settings     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  TCP 5003  CMD_DETECTION ──►  ◄── CMD_AIMOVE / CMD_AIMODE / KILL    │
│  TCP 8003  video JPEG stream ──►                                    │
└──────────────────────────────────────────────────────────────────────┘
        CMD_DETECTION ──►                    ◄── CMD_AIMOVE
        (YOLOv8 results + sonic)             (FORWARD/SLOW/STOP/REROUTE)
┌──────────────────────────────────────────────────────────────────────┐
│  Client  (operator laptop / PC — GPU recommended)                    │
│                                                                      │
│  ai_viewer.py         ← full AI client (V-JEPA 2 + SSv2 + decision) │
│  world_model.py       ← V-JEPA 2 future-latent prediction           │
│  temporal_action.py   ← SSv2-style motion pattern recognition       │
│  decision.py          ← weighted risk fusion + hysteresis           │
│  config_client.yaml   ← client AI thresholds (device, weights …)   │
│  Main.py              ← original Freenove client (unchanged)        │
└──────────────────────────────────────────────────────────────────────┘
```

### Why split this way?

| Concern | Pi 5 | Laptop / PC |
|---|---|---|
| V-JEPA 2 (ViT-L, ~300 MB) | slow on CPU-only | GPU or fast CPU |
| YOLOv8n (4 MB, runs every 2 frames) | sufficient | unnecessary round-trip |
| Ultrasonic safety stop | must be local (hard real-time) | network latency risk |
| Motor command latency | <1 ms local | ~5–20 ms LAN (acceptable for navigation) |

---

## Baseline vs Predictive comparison

| Feature | Baseline | Predictive |
|---|---|---|
| YOLOv8 detection | ✓ | ✓ |
| V-JEPA 2 future prediction | ✗ (weight = 0) | ✓ (weight = 0.45) |
| SSv2-style temporal patterns | ½ weight | full weight |
| Ultrasonic guard | ✓ | ✓ |
| V-JEPA 2 early-warning deceleration | ✗ | ✓ |

Both modes use the **same code path**. The only difference is the weight
vector in `Code/Client/config_client.yaml` under `decision.weights`.
Switching is instant — click PREDICTIVE or BASELINE in the client UI and
the `DecisionFuser` is rebuilt with the new weights on the client.

---

## Setup

### Raspberry Pi 5 (server)

```bash
# 1. Clone this repo onto the Pi
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# 2. Install Pi system packages
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-gpiozero python3-opencv

# 3. Install Python packages (YOLOv8 + Freenove deps; no torch needed here)
pip3 install -r requirements_server.txt

# 4. (First run only) Create hardware params file
cd Code/Server
python3 parameter.py
```

The Pi runs **YOLOv8 only** — no torch or transformers needed on the Pi.

### Client (operator laptop / PC)

```bash
pip install -r requirements_client.txt
```

`requirements_client.txt` includes `torch` and `transformers`.
**V-JEPA 2 weights (~300 MB) are downloaded from HuggingFace on the first
client run.** If the client has no internet access, a lightweight
`_StubEncoder` (fixed random projection of pixel statistics) is used
automatically — enough to exercise the full pipeline and show the
predictive vs baseline difference.

To use a GPU (strongly recommended):

```bash
# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch   # standard install already includes MPS

# Then set in Code/Client/config_client.yaml:
#   world_model:
#     device: "cuda"   # or "mps"
```

---

## How to run

### Prerequisites

| Machine | Required packages |
|---|---|
| Raspberry Pi 5 (server) | `python3-picamera2 python3-gpiozero python3-opencv` (apt) + `requirements_server.txt` |
| Laptop / PC (client) | `requirements_client.txt` (includes torch + transformers) |

---

### Option A – Demo mode (no robot hardware needed)

The fastest way to see predictive vs baseline. Everything runs on a single
machine — the server and client on the same laptop.

**Step 1** – Provide a corridor video clip:

```
assets/demo_clips/corridor.mp4
```

Any short indoor walkway video works (30 s is plenty). The clip loops.

**Step 2** – Start the server in demo mode (one terminal):

```bash
cd Code/Server

# Predictive navigation (V-JEPA 2 active on client)
python main_predictive.py --mode demo --nav predictive

# Baseline reactive (YOLOv8 + temporal only, no world model)
python main_predictive.py --mode demo --nav baseline

# Custom video path
python main_predictive.py --mode demo --video /path/to/my_video.mp4

# Headless (no OpenCV window on the server side)
python main_predictive.py --mode demo --no-display
```

The server window shows **YOLO bounding boxes, sonic distance, and the
last action received from the client**. V-JEPA 2 label and motion pattern
are displayed in the client UI, not the server window.

**Step 3** – Start the client viewer (second terminal):

```bash
cd Code/Client
python ai_viewer.py
# Type 127.0.0.1 in the IP field and click Connect
```

The client UI shows four risk bars (fused, YOLO from Pi, V-JEPA 2 from PC,
temporal from PC), the world model label, motion pattern, and sonic
distance. V-JEPA 2 loads in the background — the "AI: loading models…"
indicator turns green when inference is ready.

---

### Option B – Live mode (physical Freenove Tank Robot)

**Step 1** – One-time hardware parameter file (run once on the Pi):

```bash
cd Code/Server
python3 parameter.py
```

**Step 2** – Start the server on the Pi:

```bash
cd Code/Server
python main_predictive.py --mode live --nav predictive
```

The server binds on:
- TCP port **5003** — commands (`CMD_DETECTION` out, `CMD_AIMOVE` / `CMD_AIMODE` in)
- TCP port **8003** — video JPEG stream

**Step 3** – Start the client viewer on your laptop:

```bash
cd Code/Client
python ai_viewer.py
# Enter the Pi's IP address and click Connect
```

The client loads V-JEPA 2, connects to the Pi, and begins the
`CMD_DETECTION → decision → CMD_AIMOVE` control loop automatically.

---

### Client viewer controls

| Control | Action |
|---|---|
| **PREDICTIVE** button / `Ctrl+P` | Rebuild DecisionFuser with V-JEPA 2 weight = 0.45 |
| **BASELINE** button / `Ctrl+B` | Rebuild DecisionFuser with V-JEPA 2 weight = 0 |
| **STOP AI / MANUAL** button | Pause AI; no CMD_AIMOVE sent; manual CMD_MOTOR honoured |
| **EMERGENCY STOP** / `Space` / `Esc` | Send CMD_AIMODE#0 — motors halt, AI paused |
| **SHUTDOWN SERVER** / `Ctrl+Q` | Send CMD_KILL#0 — server process exits cleanly |

EMERGENCY STOP shows a confirmation message for 4 seconds then resets.
Click PREDICTIVE or BASELINE to resume after any stop.

---

### Baseline vs predictive comparison walkthrough

1. Place the robot at the corridor start.
2. Run `python main_predictive.py --mode live --nav baseline` on the Pi.
3. In the client, click **BASELINE** to confirm baseline weights.
4. Walk into the camera view from the side. Watch the **YOLO risk bar** in
   the client — note the frame at which it crosses the SLOW threshold. This
   is the **baseline reaction point**.
5. Click **PREDICTIVE** (no restart needed). Repeat the approach.
6. Watch the **V-JEPA 2 bar** and **world model label** in the client UI.
   The label should flip to `BLOCKED` and the fused risk bar should cross
   the SLOW threshold **earlier** than in baseline mode — before the person
   fills the camera frame.
7. The logged CSV in `logs_rpi/` records `detector_risk` per frame from the
   Pi. Compare across runs to see the reaction point difference.

---

### Calibrate V-JEPA 2 reference prototypes

The default synthetic prototypes (grey wall vs bright open corridor) work
for most environments. To calibrate for your specific corridor, run the
anchor builder on the **client machine** (since V-JEPA 2 runs there):

```bash
# Future: interactive anchor builder for the client
# For now, synthetic prototypes are used automatically on first run.
# They are recalibrated by collecting frames via --build-anchors on the
# server and manually passing embeddings to the client WorldModel instance.
```

For a demo the synthetic defaults are sufficient.

---

## Extended TCP protocol

New commands added on top of the standard Freenove protocol:

| Command | Direction | Format | Meaning |
|---|---|---|---|
| `CMD_DETECTION` | Pi → Client | `CMD_DETECTION#<yolo_risk_pct>#<obs_in_center>#<area_frac_pct>#<centroid_x_pct>#<sonic_cm>` | Per-frame YOLOv8 results + sonic |
| `CMD_AIMOVE` | Client → Pi | `CMD_AIMOVE#FORWARD` / `#SLOW` / `#STOP` / `#REROUTE` | Navigation action from client AI |
| `CMD_AIMODE` | Client → Pi | `CMD_AIMODE#0` / `#1` / `#2` | Stop AI / Baseline / Predictive |
| `CMD_KILL` | Client → Pi | `CMD_KILL#0` | Stop motors + shut down server process |

The `CMD_DETECTION → CMD_AIMOVE` pair is the AI control loop:

```
Pi:     detect → CMD_DETECTION (every frame, ~8 fps)
Client: V-JEPA 2 + SSv2 + decision → CMD_AIMOVE (~10 Hz)
Pi:     setMotorModel(left, right)
```

All existing Freenove commands (`CMD_MOTOR`, `CMD_SERVO`, `CMD_LED`, etc.)
continue to work. Manual `CMD_MOTOR` passes through when `CMD_AIMODE#0`
(stop AI) is active.

---

## Something-Something V2 motion patterns

The temporal recogniser classifies the **trajectory** of detections over the
last `window_size` (default 10) frames using linear regression on bounding
box area and centroid position — no neural network, runs every frame at
negligible CPU cost.

| Pattern | What the recogniser sees | Risk |
|---|---|---|
| `STATIC_CLEAR` | No detection for most frames | 0.00 |
| `CLEARING` | Detection present early, absent recently | 0.10 |
| `UNCERTAIN` | Not enough history | 0.25 |
| `CROSSING` | Centroid moving sideways, stable area | 0.45 |
| `APPROACHING` | Area growing + obstacle centred | 0.70 |
| `BLOCKING` | Large, centred, stationary | 0.85 |

---

## Running tests

```bash
# Tests that need no GPU or hardware (run on any machine)
pytest tests_rpi/ -v

# Original ESP32-targeted tests
pytest tests/ -v
```

24 tests run across 4 test files covering decision fusion, temporal
patterns, world model mathematics, and camera buffer behaviour.

---

## Configuration

### Server — `Code/Server/config.yaml`

| Setting | Default | Effect |
|---|---|---|
| `mode` | `demo` | `demo` (video file) or `live` (Pi camera) |
| `detector.model` | `yolov8n.pt` | YOLOv8 model size (n/s/m) |
| `detector.run_every_n_frames` | `2` | YOLOv8 cadence (CPU saving on Pi) |
| `robot.speed_full` | `1500` | Full-speed PWM (max 4095) |
| `robot.speed_slow` | `800` | Reduced-speed PWM |
| `robot.ultrasonic_stop_cm` | `15.0` | Hard stop distance (cm) — safety override on Pi |
| `camera.clip_length` | `16` | Frame buffer depth (must match client) |

### Client — `Code/Client/config_client.yaml`

| Setting | Default | Effect |
|---|---|---|
| `navigation_mode` | `predictive` | Starting mode on launch |
| `world_model.device` | `cpu` | `cuda` / `mps` / `cpu` for V-JEPA 2 |
| `world_model.risk_similarity_threshold` | `0.55` | V-JEPA 2 BLOCKED sensitivity |
| `world_model.run_every_n_frames` | `8` | V-JEPA 2 inference cadence |
| `decision.weights.world_model` | `0.45` | V-JEPA 2 contribution to fused risk |
| `decision.weights.detector` | `0.35` | YOLOv8 contribution |
| `decision.weights.temporal` | `0.20` | SSv2 temporal contribution |
| `decision.low_risk_max` | `0.30` | Below this → FORWARD |
| `decision.medium_risk_max` | `0.60` | Below this → SLOW, above → STOP/REROUTE |
| `camera.clip_length` | `16` | Must match server `camera.clip_length` |

---

## Logging

Each run creates a timestamped directory under `logs_rpi/`:

```
logs_rpi/
└── run_20240515_143022_predictive/
    ├── navigation_log.csv   # one row per frame: YOLO risk, action, sonic
    ├── system.log           # Python logging output
    └── frames/              # annotated JPEGs (every 5th frame)
```

The Pi-side CSV records `detector_risk` (YOLOv8) per frame. World model
and temporal risk are computed on the client and are not in the Pi CSV.
To capture all three signals, add logging to `ai_viewer.py`'s AI loop.

---

## Success criteria

- Robot reaches the goal point reliably in both modes
- Predictive mode visibly begins decelerating **earlier** than baseline mode
- The V-JEPA 2 label in the **client UI** shows `BLOCKED` before the YOLO
  bar crosses the SLOW threshold
- Motion is smoother in predictive mode (fewer full stops from a cold start)
- Pi side runs stably at ≥ 8 FPS on a Raspberry Pi 5 (YOLOv8 only)
- Client AI loop runs at ≥ 10 Hz on a modern laptop (CPU or GPU)
- All YOLO signals are logged to CSV on the Pi for post-run analysis
