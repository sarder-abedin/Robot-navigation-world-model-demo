# Freenove Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot Kit for
Raspberry Pi (FNK0077)** that uses **V-JEPA 2** as a world model to anticipate
future obstacles — not just react to what is currently visible.

---

## Project overview

The robot drives through a small indoor corridor (chairs, boxes, tape lines,
temporary occlusions, people crossing).  Instead of only stopping when an
obstacle fills the camera frame, the system uses three complementary layers:

| Layer | Model | Role | Where it runs |
|---|---|---|---|
| Instantaneous detection | **YOLOv8n** | What is blocking the path *right now*? | **Server (Pi 5)** |
| Predictive world model | **V-JEPA 2** | What will the scene look like in ~0.5 s? | **Client (laptop/PC)** |
| Temporal motion pattern | **SSv2-style rules** | Is the obstacle approaching, crossing, or clearing? | **Client (laptop/PC)** |
| Decision fusion | Weighted risk fuser | Combine all three signals → action | **Client (laptop/PC)** |
| Motor execution | **Freenove tankMotor** | `setMotorModel(L, R)` | **Server (Pi 5)** |

The Pi handles fast reactive tasks (YOLOv8, ultrasonic, motors).  The client
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
│  TCP 5003 (CMD_DETECTION ──►  CMD_AIMOVE ◄── CMD_AIMODE/KILL)       │
│  TCP 8003 (video stream JPEG ──►)                                   │
└──────────────────────────────────────────────────────────────────────┘
       CMD_DETECTION ──►                     ◄── CMD_AIMOVE
       (YOLOv8 results, sonic)               (FORWARD/SLOW/STOP/REROUTE)
┌──────────────────────────────────────────────────────────────────────┐
│  Client  (operator laptop / PC with GPU)                             │
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

| Concern | Pi 5 | Laptop/PC |
|---|---|---|
| V-JEPA 2 (ViT-L, ~300 MB) | ❌ slow on CPU-only | ✅ GPU or fast CPU |
| YOLOv8n (4 MB, 2 FPS budget) | ✅ sufficient | unnecessary round-trip |
| Ultrasonic safety | ✅ must be local (hard real-time) | ❌ network latency risk |
| Motor latency | ✅ <1 ms local | ~5–20 ms LAN (acceptable) |

---

## How V-JEPA 2 is used as a world model

V-JEPA 2 predicts **future latent representations** of a scene without
reconstructing pixels.  We exploit this to ask:

> "Given the last 16 frames, what will the scene embedding look like 4 frames
> from now (~0.5 s at 8 fps)?"

1. A **rolling 16-frame buffer** is maintained from the camera stream.
2. V-JEPA 2 runs with the last 4 frames **masked**, forcing the predictor to
   imagine the near future.
3. The predicted embedding is compared (cosine similarity) against two anchors:
   - `obstacle_anchor` — average latent of corridor frames with a centred blocker
   - `clear_anchor` — average latent of obstacle-free corridor frames
4. The similarity difference becomes `predicted_risk ∈ [0, 1]`.

**Why this matters:** a person entering at the frame edge has a low *current*
detection risk (small bbox, off-centre) but produces a *predicted* embedding
much closer to the obstacle anchor than the clear anchor.  The robot starts
decelerating several frames before the baseline system would react.

The predictive early-warning path is in `decision.py:65`:
```python
if self._mode == "predictive" and world_model_label == "BLOCKED" and action == Action.FORWARD:
    action = Action.SLOW   # decelerate proactively
```

---

## Baseline vs Predictive comparison

| Feature | Baseline | Predictive |
|---|---|---|
| YOLOv8 detection | ✓ | ✓ |
| V-JEPA 2 future prediction | ✗ (weight = 0) | ✓ (weight = 0.45) |
| SSv2-style temporal patterns | ½ weight | full weight |
| Ultrasonic guard | ✓ | ✓ |
| V-JEPA 2 early-warning deceleration | ✗ | ✓ |

Both modes run on the **same code path** — the only difference is the weight
vector in `config.yaml decision.weights`.

---

## Setup

### Raspberry Pi (server)

```bash
# 1. Clone this repo onto the Pi
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo

# 2. Install Pi system packages
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-gpiozero python3-opencv

# 3. Install Python packages
pip3 install -r requirements_server.txt

# 4. (First run only) Create hardware params file
cd Code/Server
python3 parameter.py
```

V-JEPA 2 weights (~300 MB) are downloaded from HuggingFace on first run.
If the Pi has no internet access, the system automatically falls back to a
lightweight `_StubEncoder` that produces consistent embeddings from pixel
statistics — enough to exercise the full pipeline and demonstrate the
predictive vs baseline difference.

### Client (operator laptop)

```bash
pip install -r requirements_client.txt
```

---

## How to run

### Prerequisites

| Machine | Packages |
|---|---|
| Raspberry Pi (server) | `python3-picamera2 python3-gpiozero python3-opencv` (via apt) + `requirements_server.txt` |
| Laptop / PC (client) | `requirements_client.txt` |

```bash
# Raspberry Pi
sudo apt-get install -y python3-picamera2 python3-gpiozero python3-opencv
pip3 install -r requirements_server.txt

# Laptop / PC
pip install -r requirements_client.txt
```

V-JEPA 2 weights (~300 MB) are fetched from HuggingFace automatically on
first run.  If the Pi has no internet access the system falls back to a
lightweight stub encoder that preserves the full pipeline behaviour.

---

### Option A – Demo mode (laptop only, no robot hardware needed)

This is the fastest way to see the predictive vs baseline comparison.

**Step 1** – Provide a corridor video clip:

```
assets/demo_clips/corridor.mp4
```

Any short indoor walkway video works (30 s is plenty).  The clip loops
automatically.

**Step 2** – Start the server in demo mode:

```bash
cd Code/Server

# Predictive navigation (V-JEPA 2 active)
python main_predictive.py --mode demo --nav predictive

# Baseline reactive (YOLOv8 + temporal only, no world model)
python main_predictive.py --mode demo --nav baseline

# Custom video path
python main_predictive.py --mode demo --video /path/to/my_video.mp4

# Headless (no OpenCV window)
python main_predictive.py --mode demo --no-display
```

An annotated OpenCV window opens showing detections, the risk bar, the
V-JEPA 2 world-model label, the motion pattern, and the current action.

**Step 3 (optional)** – Connect the client viewer while the server runs:

```bash
# In a second terminal (or on a second machine on the same network)
cd Code/Client
python ai_viewer.py
# Type 127.0.0.1 (localhost) in the IP field and click Connect
```

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
- TCP port **5003** (commands)
- TCP port **8003** (video stream)

**Step 3** – Connect the client viewer from your laptop:

```bash
cd Code/Client
python ai_viewer.py
# Enter the Pi's IP address and click Connect
```

---

### Client viewer controls

| Control | Action |
|---|---|
| **PREDICTIVE MODE** button / `Ctrl+P` | Switch to V-JEPA 2 predictive mode |
| **BASELINE MODE** button / `Ctrl+B` | Switch to reactive-only baseline mode |
| **STOP AI** button | Pause AI control; manual `CMD_MOTOR` commands honoured |
| **EMERGENCY STOP** button / `Space` / `Esc` | Immediately cut motor power and pause AI |
| **SHUTDOWN SERVER** button / `Ctrl+Q` | Send `CMD_KILL` – stop motors, shut down the server process |

The **EMERGENCY STOP** button turns green for 4 seconds to confirm the
command was sent, then resets to ready state.  To resume driving after an
emergency stop, click **PREDICTIVE MODE** or **BASELINE MODE**.

---

### Calibrate V-JEPA 2 anchors for your corridor

The default anchors work for most indoor corridors.  To recalibrate for
your specific environment:

```bash
cd Code/Server
python main_predictive.py --build-anchors
# o  → label current frame as "obstacle"
# c  → label current frame as "clear"
# q  → finish and save anchors
```

At least 10 frames of each class are recommended.

---

### Baseline vs predictive comparison walkthrough

1. Place the robot at the corridor start.
2. Run `python main_predictive.py --mode live --nav baseline` (or demo).
3. Walk into the camera view from the side.  Note the frame at which the
   robot begins to decelerate — this is the **baseline reaction point**.
4. Restart with `--nav predictive`.
5. Repeat the same approach.  V-JEPA 2 should raise the world-model label
   to `BLOCKED` (visible in the HUD and client status bar) **before** the
   YOLOv8 bounding box fills the risk bar.  The robot decelerates earlier.
6. Compare the logged CSVs in `logs_rpi/` — the `world_model_risk` column
   rises several frames ahead of `detector_risk` in predictive mode.

---

### Demo mode (recorded video – quick reference)

```bash
cd Code/Server

# Predictive mode
python main_predictive.py --mode demo --nav predictive

# Baseline reactive mode (for comparison)
python main_predictive.py --mode demo --nav baseline

# Custom video file
python main_predictive.py --mode demo --video /path/to/my_video.mp4

# Disable display window (headless)
python main_predictive.py --mode demo --no-display
```

### Calibrate V-JEPA 2 anchors for your corridor

```bash
cd Code/Server
python main_predictive.py --build-anchors
# Press 'o' to label obstacle frames, 'c' for clear frames, 'q' to finish
```

---

## Extended TCP protocol

This project adds new commands on top of the standard Freenove protocol:

| Command | Direction | Format | Meaning |
|---|---|---|---|
| `CMD_AIMODE` | client → server | `CMD_AIMODE#0` / `#1` / `#2` | Stop AI / Baseline / Predictive |
| `CMD_AIMOVE` | client → server | `CMD_AIMOVE#FORWARD` / `#SLOW` / `#STOP` / `#REROUTE` | AI navigation action (from client V-JEPA2+SSv2+decision) |
| `CMD_KILL` | client → server | `CMD_KILL#0` | Stop motors + shut down server process |
| `CMD_DETECTION` | server → client | `CMD_DETECTION#<yolo_risk_pct>#<obs_in_center>#<area_frac_pct>#<centroid_x_pct>#<sonic_cm>` | Per-frame YOLOv8 detection results |

All existing Freenove commands (`CMD_MOTOR`, `CMD_SERVO`, `CMD_LED`, etc.)
continue to work as before.  Manual `CMD_MOTOR` commands are honoured when
`CMD_AIMODE#0` (stop AI) is active.

The `CMD_DETECTION` → `CMD_AIMOVE` round-trip is the new AI control loop:
the Pi detects obstacles and sends results to the client; the client runs
V-JEPA 2 + SSv2 + decision fusion and replies with a navigation action that
the Pi executes on its motors.

---

## Something-Something V2 motion patterns

The temporal recogniser classifies the **trajectory** of detections over the
last `window_size` (default 10) frames into SSv2-inspired categories:

| Pattern | Description | Risk |
|---|---|---|
| `STATIC_CLEAR` | No obstacle in the window | 0.00 |
| `CLEARING` | Obstacle was present but is moving away | 0.10 |
| `UNCERTAIN` | Not enough signal | 0.25 |
| `CROSSING` | Obstacle moves laterally without growing | 0.45 |
| `APPROACHING` | Area growing, obstacle in centre zone | 0.70 |
| `BLOCKING` | Large, centred, stationary obstacle | 0.85 |

---

## Running tests

```bash
# Tests that need no GPU or hardware (run on any machine)
pytest tests_rpi/ -v

# Original ESP32-targeted tests
pytest tests/ -v
```

24 tests run across 4 test files, covering decision fusion, temporal patterns,
world model mathematics, and camera buffer behaviour.

---

## Configuration (`Code/Server/config.yaml`)

Key knobs for the demo:

| Setting | Default | Effect |
|---|---|---|
| `navigation_mode` | `predictive` | Which mode starts on launch |
| `world_model.risk_similarity_threshold` | `0.55` | V-JEPA 2 BLOCKED sensitivity |
| `decision.weights.world_model` | `0.45` | V-JEPA 2 contribution to fused risk |
| `decision.low_risk_max` | `0.30` | Below this → FORWARD |
| `decision.medium_risk_max` | `0.60` | Below this → SLOW, above → STOP/REROUTE |
| `robot.ultrasonic_stop_cm` | `15.0` | Hard stop distance (cm) |
| `robot.speed_full` | `1500` | Full-speed PWM (max 4095) |
| `robot.speed_slow` | `800` | Reduced-speed PWM |
| `detector.run_every_n_frames` | `2` | YOLOv8 cadence (CPU saving) |
| `world_model.run_every_n_frames` | `8` | V-JEPA 2 cadence (CPU saving) |

---

## Logging

Each run creates a timestamped directory under `logs_rpi/`:

```
logs_rpi/
└── run_20240515_143022_predictive/
    ├── navigation_log.csv   # one row per frame: action, risk scores, labels
    ├── system.log           # Python logging output
    └── frames/              # annotated JPEGs (every 5th frame)
```

The CSV captures all three risk signals separately, so you can plot predictive
vs baseline risk trajectories from the same scenario and show how V-JEPA 2
raises risk earlier.

---

## Success criteria

- Robot reaches the goal point reliably in both modes
- Predictive mode visibly begins decelerating **earlier** than baseline mode
- The V-JEPA 2 label shows `BLOCKED` before the YOLOv8 detector fills the risk bar
- Motion is smoother in predictive mode (fewer full stops from a cold start)
- System runs stably at ≥ 8 FPS on a Raspberry Pi 4 (demo mode)
- All signals are logged to CSV for post-run analysis
