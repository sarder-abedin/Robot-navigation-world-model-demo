# Architecture — Tank Robot Predictive Navigation

> Part of the [Tank Robot](README.md) docs — see also [ARCHITECTURE.md](ARCHITECTURE.md) · [HOW_TO_RUN.md](HOW_TO_RUN.md) · [CALIBRATION.md](CALIBRATION.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

How the system works under the hood: the split-inference topology, what each AI
model contributes, the safety layers, and the TCP wire protocol.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PC / Laptop  (TCP SERVER – runs ALL AI)    [venv OR Docker]    │
│                                                                 │
│  main_server.py              ← entry point                      │
│  ├── YOLO11n                 ← object detection (all modes)      │
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
- **All AI runs on the PC** — YOLO11n object detection, V-JEPA 2, SSv2 and the
  decision fuser (GPU-capable heavy inference). The Pi runs no AI.
- The Pi is a **thin client**: it streams JPEG camera frames to the PC and sends
  `CMD_SONIC` (its ultrasonic reading, the local hard-stop safety); the PC runs
  YOLO on those frames.
- **Fail-safe link handling on the Pi:** a **motor watchdog** stops the motors if
  no command arrives from the PC within `robot.command_watchdog_seconds` (default
  1.5 s) — covering a stalled pipeline/server, a stalled video stream, or a
  silently half-open link. TCP **keepalive** on both sockets tears down a truly
  dead connection, and an **outer reconnect loop** drops the robot back to
  (re)connecting (motors stopped) instead of exiting on a transient network blip.

---

## What each AI model does

| Model | Nickname | What it does | Where it runs |
|---|---|---|---|
| **YOLO11n** | "The Photographer" | Spots obstacles in the current frame; produces aggregated risk+position **and the largest obstacle's class label** | PC |
| **V-JEPA 2** | "The Fortune Teller" | Predicts what the scene will look like 0.5 s from now in latent space | PC |
| **Depth-Anything V2** | "The Surveyor" | Class-agnostic depth → free-space distance ahead + which side is open (sees walls YOLO can't); feeds the governor + REROUTE direction | PC (optional) |
| **SSv2 temporal rules** | "The Behaviour Analyst" | Classifies the obstacle's motion pattern (APPROACHING / CROSSING / BLOCKING …) — drives `temporal_risk` | PC |
| **SSv2 model (VideoMAE)** | "The Narrator" | A **real** Something-Something-V2 video classifier; its "something" slot is filled with YOLO's object → e.g. *"person moving closer"*. Annotation/log only | PC |
| **Decision fuser** | "The Judge" | Combines all risk signals into one action (FORWARD / SLOW / STOP / REROUTE / BACKUP) + closed-loop, context-aware avoidance | PC |

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
- Runs every `ssv2.run_every_n_frames` (default 32) on CPU, auto-halved on a GPU;
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
- **During the run:** the **"Run Logging" toggle** — in the Streamlit UI *and*
  the PyQt `ai_viewer.py` — sends `CMD_LOGGING#<0|1>`.

The CSV includes an `ssv2` column with the composed sentence, plus per-frame
**inference latency** columns (`lat_total_ms` and per stage: `lat_yolo_ms`,
`lat_wm_ms`, `lat_depth_ms`, `lat_temporal_ms`, `lat_ssv2_ms`, `lat_decision_ms`,
and the governor's `reaction_ema_ms`) and camera-stream **network statistics**
(`net_recv_fps`, `net_frame_bytes`, `net_frames_recv`, `net_frames_dropped`,
`net_kbps`). The heavy models run every N frames, so their per-frame latency is
near-0 on skipped frames and spikes on the compute tick — that periodicity is
visible in the log.

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

## Proactive collision avoidance — the speed governor

On top of the risk logic, a **kinematic safe-speed governor**
(`Code/Server/speed_governor.py`) makes the robot **never travel faster than it
can stop** within the confirmed-clear distance ahead — accounting for how long
the AI takes to react. It uses the driver's-ed *total stopping distance* model:

```
d_stop(v) = v · t_react  +  (v² − v_target²) / (2·a)  +  margin
            └ thinking ┘    └──── braking ─────────┘    └ buffer ┘
```

Each cycle it picks the **fastest action that still fits**: if the clear distance
`d ≥ d_stop(v_forward)` it allows FORWARD; else if `d ≥ d_stop(v_slow)` it caps to
SLOW; else STOP. It only ever slows the robot down, and is skipped when the
ultrasonic is blind (the hard-stop / blind-hold still apply).

Why it matters:
- **Proactive** — speed is a smooth function of distance, so it slows *early*, not at the last cm.
- **Latency-aware** — the `v · t_react` term reserves braking room for the AI's decision time. `t_react` is measured live (EMA of per-frame processing), so a slow **V-JEPA 2 on CPU** automatically forces a lower safe speed instead of causing a crash.
- **Reduce impact** — set `target_speed_mps > 0` to brake to a crawl rather than requiring a dead stop.

> Calibrating the governor's SI constants on the real robot is covered in
> [HOW_TO_RUN.md → Calibrating the speed governor](HOW_TO_RUN.md#calibrating-the-speed-governor).

---

## Depth free-space channel — genuinely "seeing" the wall

YOLO only knows its 80 classes and V-JEPA 2 gives a single global risk scalar —
neither can say *"there's a wall 0.4 m ahead and the left is open."* The optional
**Depth-Anything V2** channel (`Code/Server/depth_perception.py`) adds
**class-agnostic geometry**: a per-pixel depth map for *any* scene (walls, doors,
furniture), from which the pipeline derives:

- **`clear_distance_m`** — the nearest obstacle straight ahead (metric), fed to
  the **speed governor** as the *more conservative* of {ultrasonic, depth}. This
  catches the angled/soft walls the ultrasonic misses.
- **`clear_direction`** (LEFT/CENTER/RIGHT) — the most open third, which gives
  **REROUTE an actual direction** to turn toward (`CMD_AIMOVE#REROUTE#LEFT|RIGHT`).

Combined with YOLO it "understands" obstacles: geometry from depth + a **label**
from YOLO's closest object (`… SCENE: chair 0.42m ahead | open=LEFT`). The
operator HUD shows this live too — a **"Depth: 0.42m ahead | open: LEFT"** line
with LEFT/CENTER/RIGHT free-space bars (the open side highlighted). Walls with
no YOLO class are still seen as `obstacle/wall`. It falls back to **off** when the
model/weights are absent (ultrasonic + V-JEPA 2 keep working). Configure under
`depth:` in `config.yaml`; a GPU/MPS box is recommended.

---

## Dynamic reroute — closed-loop, context-aware avoidance

Instead of a fixed back-up-then-spin, a high-risk obstacle now selects a
**behaviour** from motion + object + geometry, and turns are **closed-loop**
(`decision.py`, `decision.reroute.closed_loop`, default on):

| Situation | Signals | Action |
|---|---|---|
| Crossing / leaving obstacle, or a **person** approaching (not close) | temporal pattern `CROSSING`/`CLEARING`; YOLO class ∈ `dynamic_classes` | **WAIT** (`STOP`) — the path may clear itself; a timeout escalates to TURN |
| Obstacle **rushing in**, too close to turn | `APPROACHING` + depth center < `backup_distance_m` | **BACKUP** (`CMD_AIMOVE#BACKUP`) — capped (no rear sensor) then STOP |
| Static blockage / wall, a **side clearly more open** than centre | a side beats centre by `direction_margin_m` (absolute) **or** `direction_margin_frac` (relative) | **TURN** toward the open side (per-side depth), **keep turning until the gap opens** (a spin guard stops an endless rotation) |
| Boxed in — **no side clearly more open** than centre | neither margin met | **STOP** briefly, then **rotate in place to SEARCH** for an opening (capped by the spin guard) — the robot never just freezes facing a wall |

> The turn margin is tested both **absolutely and relatively** so it works on
> **uncalibrated** depth, where the per-side distances differ by only a few
> centimetres/percent. An absolute-only margin (the old 0.3 m) is unreachable for
> a depth camera, so the robot would sit in `STOP` and never reroute.

The **direction** comes from the per-side depth free-space (`depth_left/right`),
not a coarse hint; turning is re-evaluated every frame, so it stops the instant
the center clears. The motion signal is the **fast per-frame temporal pattern**
(SSv2's heavy VideoMAE stays annotation-only). Guards: `wait_timeout_seconds`,
`max_turn_seconds`, `max_backup_seconds`. Set `closed_loop: false` for the legacy
one-shot reroute. The ultrasonic hard-stop and speed governor still sit
underneath and can only make the action *more* cautious.

**Two supporting fixes make this respond to walls** (which YOLO can't class):
- The **motion recogniser is depth-aware** — when YOLO sees no box but depth
  shows an obstacle within `temporal_action.depth_presence_range_m`, it feeds a
  synthetic "present, centred" state so a wall registers as `APPROACHING`/
  `BLOCKING` instead of being stuck at `STATIC_CLEAR`.
- The **V-JEPA 2 label is relative** (`BLOCKED` iff obstacle-similarity exceeds
  clear-similarity by `world_model.label_margin`), so it no longer sticks on
  `BLOCKED` with uncalibrated/synthetic anchors — **calibrate the anchors** for
  the label + risk to be meaningful.

**The ultrasonic hard-stop also drives reroute.** A bare wall the *sonar* sees
often never raises the *vision* risk (YOLO can't class it → `det=0`; motion
`STATIC_CLEAR` → `ta=0`), so the vision path alone would sit stopped forever. The
ultrasonic hard-stop is an immediate reflex `STOP`, but if the obstacle won't
clear within `ultrasonic_escalate_seconds` it **escalates into the same avoidance
maneuver** (turn/back-up/rotate-to-search) to go around it. Once maneuvering, a
**hysteresis** keeps the robot committed until the front is clear by a margin
(`ultrasonic_resume_risk`) before resuming `FORWARD` — otherwise a momentary
clear (the back-up phase, or the obstacle grazing the threshold) flips it to
`FORWARD` and it drives straight back in, oscillating forward/backward. On the Pi
a continuous stream of `REROUTE` frames does **not** restart the back-up+spin each
frame (that would trap it in the back-up phase); the in-progress maneuver runs to
completion.

---

## Baseline vs Predictive comparison

| Feature | Baseline | Predictive |
|---|---|---|
| YOLO11 detection (on PC) | ✓ | ✓ |
| V-JEPA 2 future prediction | ✗ (weight = 0) | ✓ (weight = 0.45) |
| SSv2 temporal patterns | ½ weight | full weight |
| Ultrasonic guard | ✓ | ✓ |
| V-JEPA 2 early-warning deceleration | ✗ | ✓ |

Both modes run on the **same code path** — only the weight vector changes.

---

## TCP protocol

| Command | Direction | Format | Meaning |
|---|---|---|---|
| `CMD_SONIC` | Pi → PC | `CMD_SONIC#<sonic_cm>` | Ultrasonic distance (the Pi's local hard-stop safety); the Pi runs no detection |
| `CMD_AIMOVE` | PC → Pi | `CMD_AIMOVE#<FORWARD\|SLOW\|STOP\|REROUTE\|BACKUP>[#<LEFT\|RIGHT>]` | AI-computed action; Pi maps to motor PWM. REROUTE carries a turn direction (depth's open side); BACKUP is a short reverse pulse |
| `CMD_MOTOR` | UI → PC → Pi | `CMD_MOTOR#<L>#<R>` | Manual motor command relayed through PC |
| `CMD_STOP` | PC → Pi | `CMD_STOP` | Emergency halt (hard safety) |
| `CMD_KILL` | PC → Pi | `CMD_KILL` | Shutdown robot process |
| `CMD_AIMODE` | UI → PC | `CMD_AIMODE#<0/1/2>` | Mode change from operator |
| `CMD_LOGGING` | UI → PC | `CMD_LOGGING#<0/1>` | Toggle PC-side run logging |
| `CMD_GOAL` | UI → PC | `CMD_GOAL#<x‰>#<y‰>` | Set navigation goal at normalized image coords ×1000 (per-mille, since the parser is integer-only). **Phase 2: the point is tracked (CSRT/template) and its bearing + depth drawn on the HUD; still no motion** |
| `CMD_GOAL_CLEAR` | UI → PC | `CMD_GOAL_CLEAR` | Clear the navigation goal |
| `CMD_KILL` | UI → PC | `CMD_KILL#0` | Shutdown from operator |
| `CMD_AISTATUS` | PC → UI | `CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>#<ssv2_sentence>#<clear_dist_m>#<clear_dir>#<goal_status>` | Live AI state for the UI panel. `goal_status` ∈ none/tracking/lost/reached; on **reached** the PC stops the robot until a new UI command. Trailing fields optional for old clients |
| Video frames | Pi → PC | 4-byte LE uint32 length + JPEG | Camera stream for V-JEPA 2 (port 8004) |
| Video frames | PC → UI | 4-byte LE uint32 length + JPEG | Annotated frames (port 8003) |

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
│   │   ├── detector.py           ← YOLO11n (runs on the PC in all modes)
│   │   ├── world_model.py        ← V-JEPA 2
│   │   ├── temporal_action.py    ← SSv2-style motion-pattern heuristic (drives temporal_risk)
│   │   ├── ssv2_model.py         ← genuine SSv2 model (VideoMAE); YOLO-filled sentence for annotation/log
│   │   ├── decision.py           ← risk fusion + hysteresis + safety layers
│   │   ├── speed_governor.py     ← kinematic safe-speed governor (proactive, latency-aware)
│   │   ├── depth_perception.py   ← Depth-Anything free-space + clear direction (class-agnostic)
│   │   ├── calibrate_anchors.py  ← build V-JEPA 2 corridor anchors from frames
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
│   │   ├── calibrate_governor.py ← measure the governor's m/s constants on the robot
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
