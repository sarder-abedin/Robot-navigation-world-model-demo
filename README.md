# Freenove Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot** that uses
**V-JEPA 2** as a world model to anticipate future obstacles — not just react to
what is currently visible.

---

## Project overview

The robot drives through a small indoor corridor (chairs, boxes, tape lines,
temporary occlusions, people crossing).  Instead of only reacting when an
obstacle fills the camera frame, the system uses three complementary perception
layers:

| Layer | Model | Role |
|---|---|---|
| Instantaneous detection | **YOLOv8** | What is visible right now? |
| Predictive world model | **V-JEPA 2** | What will the scene look like in ~0.5 s? |
| Temporal motion pattern | **SSv2-style rules** | Is the obstacle approaching, crossing, blocking, or clearing? |

All three signals are fused into a single risk score that drives one of four
actions: **FORWARD → SLOW → STOP → REROUTE**.

Two modes are provided for direct comparison:

- **Predictive mode** – V-JEPA 2 and temporal reasoning are both active.
  The robot starts decelerating *before* the obstacle fully enters the frame.
- **Baseline mode** – V-JEPA 2 weight is zeroed; the robot reacts only to
  current-frame detections.

---

## Architecture

```
main.py
│
├── camera.py          LiveCamera | DemoCamera
├── detector.py        YOLOv8 → DetectionResult (instantaneous risk)
├── world_model.py     V-JEPA 2 → WorldModelResult (predictive risk)
├── temporal_action.py SSv2-style → TemporalResult (motion pattern risk)
├── decision.py        weighted fusion + hysteresis → Action
├── robot.py           RobotController | MockRobot (Freenove motor commands)
├── logger.py          CSV + annotated frame archive
├── visualization.py   OpenCV HUD overlay
└── config.yaml        all thresholds and runtime settings
```

---

## How V-JEPA 2 is used as a world model

V-JEPA 2 is a joint-embedding predictive architecture.  Its encoder maps
video clips to latent space, and its predictor imagines future latents *without
reconstructing pixels*.

In this system:

1. The last `clip_length` (default: 16) frames are kept in a rolling buffer.
2. V-JEPA 2 is run with the final `prediction_horizon` frames **masked out**,
   forcing the predictor to imagine the near future.
3. The resulting predicted embedding is compared (cosine similarity) against
   two **anchor embeddings**:
   - `obstacle_anchor` – average latent of corridor scenes with a large
     centered obstacle
   - `clear_anchor` – average latent of obstacle-free corridor scenes
4. The difference in similarity scores is converted to a `predicted_risk`
   value in [0, 1].

This means the system can see a person walking towards the frame *at the
edge*, predict that they will be blocking in ~0.5 s, and begin slowing down —
well before the baseline reactive system would react.

### Anchor calibration

The system ships with **synthetic anchors** that are good enough for a first
run.  For best results in your specific corridor, run:

```bash
python main.py --build-anchors
```

Then press `o` / `c` to label obstacle / clear frames from live camera and the
anchors are updated automatically.

---

## Setup

### Requirements

- Python 3.10+
- Linux / macOS (Windows should work but is untested with the serial port)
- For live mode: Freenove Tank Robot Kit for ESP32 connected via USB

### Install

```bash
git clone https://github.com/sarder-abedin/robot-navigation-world-model-demo
cd robot-navigation-world-model-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

V-JEPA 2 weights (~300 MB) are downloaded automatically from HuggingFace on
first run.  If no GPU is available the system falls back to a lightweight
embedding stub so the full pipeline still runs on CPU.

---

## Running the demo

### Demo mode (recorded video, no hardware required)

Place a corridor video at `assets/demo_video/corridor.mp4` then:

```bash
# Predictive mode (default)
python main.py --mode demo --nav predictive

# Baseline reactive mode (for comparison)
python main.py --mode demo --nav baseline

# Use a custom video file
python main.py --mode demo --video /path/to/my_video.mp4

# Headless (no display window)
python main.py --mode demo --no-display
```

### Live mode (physical Freenove robot)

```bash
# Check the serial port in config.yaml (robot.port), then:
python main.py --mode live --nav predictive
```

Press **q** in the display window to quit gracefully at any time.

---

## Configuration

All thresholds and settings live in `config.yaml`.

Key knobs:

| Setting | Default | Effect |
|---|---|---|
| `world_model.clip_length` | 16 | Frames in the rolling V-JEPA 2 buffer |
| `world_model.prediction_horizon` | 4 | Future frames to predict |
| `world_model.risk_similarity_threshold` | 0.55 | Cosine similarity for BLOCKED label |
| `decision.weights.world_model` | 0.45 | V-JEPA 2 contribution to fused risk |
| `decision.low_risk_max` | 0.30 | Below this → FORWARD |
| `decision.medium_risk_max` | 0.60 | Below this → SLOW, above → STOP/REROUTE |
| `decision.hysteresis` | 0.05 | Prevents oscillation near thresholds |
| `robot.speed_full` | 1800 | Full-speed PWM value |
| `robot.speed_slow` | 900 | Reduced-speed PWM value |

---

## Running tests

```bash
pytest tests/ -v
```

The test suite runs entirely without GPU or hardware — the world model tests
use the lightweight `_StubEncoder` fallback.

---

## Visualisation HUD

The live display window shows:

- **Bounding boxes** – colour-coded by obstacle class
- **Risk bar** – green → yellow → red gradient showing fused risk
- **Action label** – large centred text: FORWARD / SLOW / STOP / REROUTE
- **V-JEPA 2 label** – CLEAR / MIXED / BLOCKED prediction
- **Motion pattern** – STATIC_CLEAR / APPROACHING / CROSSING / BLOCKING / CLEARING
- **Mode badge** – PREDICTIVE (green) or BASELINE (amber)
- **FPS counter**

---

## Logging

Each run creates a timestamped directory under `logs/`:

```
logs/
└── run_20240515_143022_predictive/
    ├── navigation_log.csv      # one row per frame
    ├── system.log              # text log
    └── frames/                 # annotated JPEGs (every 5th frame)
```

The CSV captures: timestamp, action, risk scores from all three layers,
detection counts, and the explanation string.

---

## Something-Something V2 motion patterns

The temporal recogniser classifies the **trajectory** of detections over the
last `window_size` (default: 10) frames into SSv2-inspired categories:

| Pattern | Description | Risk |
|---|---|---|
| `STATIC_CLEAR` | No obstacle in the window | 0.00 |
| `CLEARING` | Obstacle was present but is moving away | 0.10 |
| `CROSSING` | Obstacle moves laterally without growing | 0.45 |
| `UNCERTAIN` | Not enough signal | 0.25 |
| `APPROACHING` | Area growing, obstacle in center | 0.70 |
| `BLOCKING` | Large, centered, stationary obstacle | 0.85 |

---

## Hardware wiring (Freenove)

The system sends `CMD_MOTOR#<lf>#<lb>#<rf>#<rb>#` strings over the serial port.
This matches the Freenove ESP32 firmware protocol.

Default serial port: `/dev/ttyUSB0` (Linux).  Change `robot.port` in
`config.yaml` if your port differs.

---

## Success criteria

- Robot reaches the goal point reliably in both modes
- **Predictive mode** begins decelerating visibly earlier than baseline mode
- System runs stably at ≥ 10 FPS on a modern laptop CPU (demo mode)
- All detections, predictions, and actions are logged to CSV for post-analysis
