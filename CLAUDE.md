# CLAUDE.md – Development Guide for AI Assistants

## Project overview

Predictive indoor navigation system for the Freenove FNK0077 Tank Robot.
Split-inference architecture: Pi runs fast AI (YOLO11n), PC runs heavy AI
(V-JEPA 2 + SSv2 + decision fusion), UI viewer runs on any laptop.

## Repository layout

```
Code/Server/    ← PC AI server (V-JEPA 2, SSv2, decision, TCP server)
Code/Robot/     ← Raspberry Pi client (YOLO11n, camera, motors, ultrasonic)
Code/Client/    ← Operator UI viewers (display-only, NO AI): Streamlit browser,
                  desktop_viewer.py (native window via pywebview), PyQt5 ai_viewer.py
tests_rpi/      ← Unit tests (no GPU/hardware required)
assets/         ← Demo video clips
```

## Architecture rules

- **PC = TCP server** (binds ports 5003/8003 for UI, 5004/8004 for robot)
- **Pi = TCP client** (connects outbound; **thin client — runs NO AI**)
- **UI viewer = TCP client** (connects to PC only)
- **ALL AI runs on the PC.** YOLO11n (`Code/Server/detector.py`), V-JEPA 2 and
  SSv2 all run in `Code/Server/` in every mode (live and demo). The Pi streams
  camera frames (port 8004) and the PC detects on them. There is no on-Pi YOLO
  and no `DETECTOR_LOCATION` switch. The Pi's only sensor report is the ultrasonic
  distance (`CMD_SONIC`), its local hard-stop safety.
- The Pi Docker image is therefore lightweight — **no torch / ultralytics**.
- Motor mapping (action → PWM) happens on the **Pi** side in `main_robot.py`,
  not on the PC. The PC sends high-level `CMD_AIMOVE#FORWARD` etc.
- **Navigation safety layers (in `decision.py`, in order):** (1) ultrasonic
  **hard-stop** — deterministic, distance-only, *separate* from the fused AI
  risk; a persistent block (obstacle won't clear within
  `reroute.ultrasonic_escalate_seconds`) escalates into the closed-loop avoidance
  so a sonar-seen wall (which never raises the vision risk) is rerouted around,
  with `reroute.ultrasonic_resume_risk` hysteresis to avoid forward/backward
  oscillation; (2) **vision** action from fused det+wm+temporal risk; on high risk a
  **closed-loop, context-aware avoidance** picks WAIT (crossing/person — path may
  clear) / TURN-toward-open-side-until-clear / BACKUP (rushing in) — direction
  from per-side depth, motion from the fast temporal pattern; guards:
  `wait_timeout`/`max_turn`/`max_backup`; `decision.reroute.closed_loop: false`
  falls back to the legacy one-shot reroute; (3) **speed governor**
  (`speed_governor.py`) — caps FORWARD/SLOW so the robot can stop within the
  clear distance given the measured reaction latency. Each layer only ever makes
  the action **more** cautious. Governor speeds are SI m/s and must be calibrated.
- **Depth free-space** (`depth_perception.py`, optional Depth-Anything V2): gives
  the governor a metric `clear_distance_m` (fused as the *nearer* of ultrasonic +
  depth, so it catches angled walls the sonar misses) and a `clear_direction`
  (LEFT/CENTER/RIGHT) that supplies the REROUTE turn direction. Class-agnostic —
  combined with YOLO's `closest_label` for "obstacle 0.4m ahead (chair)". Degrades
  to off (stub) when transformers/weights are absent.
- **V-JEPA 2 anchors** default to synthetic; calibrate real ones with
  `calibrate_anchors.py` and set `world_model.anchors_path`.

## Key files

| File | Purpose |
|---|---|
| `Code/Server/main_server.py` | PC entry point; starts AI pipeline + TCP servers |
| `Code/Server/detector.py` | YOLO11n — runs on the **PC** on the Pi's streamed frames (all modes) |
| `Code/Server/robot_connection.py` | Parses `CMD_SONIC` from Pi; exposes `get_sonic_cm()` and `send_aimove()` |
| `Code/Server/ai_pipeline.py` | Orchestrates YOLO → V-JEPA 2 → SSv2 → decision → broadcast; measures reaction latency for the governor |
| `Code/Server/decision.py` | Risk fusion + hysteresis; ultrasonic hard-stop (separate); closed-loop context-aware reroute (wait/turn/backup); applies the speed governor |
| `Code/Server/speed_governor.py` | Kinematic safe-speed governor: caps action to `d_stop(v)=v·t_react+v²/(2a)+margin` |
| `Code/Server/depth_perception.py` | Depth-Anything V2 → free-space distance ahead + clear direction (LEFT/CENTER/RIGHT); class-agnostic (sees walls); `depth_at_norm()` per-pixel goal-depth sampler |
| `Code/Server/goal_navigator.py` | Tracks a user-selected goal (CSRT, template-match fallback) → bearing + depth; `goal_steering()` turns the safety decision into a goal-directed TURN/FORWARD in **Goal-Following** mode (safety always overrides). Server starts **idle**; `--ai-start` opts headless into driving |
| `Code/Server/calibrate_anchors.py` | Builds V-JEPA 2 corridor anchors from blocked/clear frame folders → `anchors.npz` |
| `Code/Server/calibrate_from_logs.py` | **Zero-driving** calibration from stored `logs_rpi/<run>/` (one or more, pooled): derives `depth.scale` + governor speeds (sonar as ground-truth ruler) and, with `--anchors`, auto-labels raw frames → anchors; `--apply` patches `config.yaml` (absolute anchors path) + verifies |
| `Code/Server/calibration_ui.py` | **Separate desk-only PyQt5 UI** for the above (`python calibration_ui.py`): pick runs → Analyze pooled values → Apply to config (anchor build off-thread; never drives the robot) |
| `Code/Server/run_report.py` | Offline matplotlib plots + summary of one run's `navigation_log.csv` (risk/distance/action/latency/network); `save_pngs()` → `<run>/viz/`. Backend-agnostic + unit-tested |
| `Code/Server/run_visualizer.py` | **Desk-only PyQt5 run visualizer** (`python run_visualizer.py`): pick a run → embedded plots + a synced annotated-frame scrubber → Save PNGs |
| `Code/Robot/calibrate_governor.py` | On-robot: measures governor m/s constants (sonar+motors), safely patches `config.yaml` |
| `Code/Server/visualization.py` | HUD overlay: keeps spatial cues on the video (boxes, risk bar, action, depth L/C/R bars, goal marker+arrow+readout); text overlays (V-JEPA2/motion, sonic, fps, ssv2, depth-distance) default OFF and are shown in the UI panel below the video instead |
| `Code/Server/robot_control.py` | `TCPRobotController` sends `CMD_AIMOVE` to Pi; direct `RobotController` reroute/backup run as preemptible worker threads (never block the pipeline); ultrasonic risk (fail-safe on no-echo) |
| `Code/Robot/main_robot.py` | Pi entry point (thin client); camera stream + sonic + command loop; motor **watchdog** (stop on PC silence) + **reconnect** loop |
| `Code/Robot/tcp_robot_client.py` | Pi-side TCP client; `send_sonic()` / `send_frame()`; TCP keepalive on both sockets |
| `Code/Client/streamlit_viewer.py` | Browser UI (port 8501); bundled in the server Docker image |
| `Code/Client/desktop_viewer.py` | Native desktop window wrapping the Streamlit UI (pywebview); native-only |
| `Code/Client/ai_viewer.py` | PyQt5 UI; on connect the operator **picks a Navigation Mode first** (Obstacle Avoidance / Goal Following) then explicitly activates; MANUAL driving available pre-pick; run-logging toggle (`CMD_LOGGING`) |

## TCP protocol (summary)

```
Pi → PC:  CMD_SONIC#<sonic_cm>                        (ultrasonic distance; local hard-stop)
Pi → PC:  4-byte LE uint32 + JPEG  (camera stream, port 8004, for ALL server-side AI)
          (the server also still accepts legacy CMD_DETECTION for backward compat)
PC → Pi:  CMD_AIMOVE#<FORWARD|SLOW|STOP|REROUTE|BACKUP|TURN>[#<LEFT|RIGHT>]  (AI action; REROUTE=back-up+spin, BACKUP=short reverse, TURN=in-place spin toward goal (Goal-Following, no backup); LEFT/RIGHT on REROUTE/TURN)
PC → Pi:  CMD_MOTOR#<L>#<R>                         (manual from UI viewer)
PC → Pi:  CMD_STOP / CMD_KILL / CMD_AIMODE#<0|1|2>
UI → PC:  CMD_AIMODE#<0|1|2>  |  CMD_MOTOR#<L>#<R>  |  CMD_KILL#0
UI → PC:  CMD_LOGGING#<0|1>                         (toggle server-side run logging)
UI → PC:  CMD_GOAL#<x_permille>#<y_permille>        (set goal at normalized image coords ×1000)
UI → PC:  CMD_GOAL_CLEAR                            (clear the goal)
UI → PC:  CMD_GOALFOLLOW#<0|1>                      (nav mode: 1=Goal Following (steer to goal), 0=Obstacle Avoidance)
PC → UI:  CMD_AISTATUS#<action>#<risk_pct>#<wm_label>#<pattern>#<sonic_cm>#<ssv2_sentence>#<clear_dist_m>#<clear_dir>#<goal_status>
          (ssv2 + depth + goal_status appended for the UI panel; goal_status ∈ none|tracking|lost|reached; on `reached` the PC stops the robot and waits for a new UI command; trailing fields optional for old clients)
PC → UI:  4-byte LE uint32 + JPEG  (annotated HUD frames, port 8003)
```

## SSv2 (genuine model) + run logging

- `Code/Server/ssv2_model.py` runs a **real** Something-Something-V2 video
  classifier (VideoMAE, `MCG-NJU/videomae-base-finetuned-ssv2`) over the clip
  buffer. The predicted template's "something" slot is filled with the largest
  obstacle's YOLO class (from the PC's own detection on the streamed frame), e.g.
  "person moving closer"; when YOLO has no named object it falls back to
  `ssv2.unknown_object_label` (default "obstacle") so the caption never leaks the
  raw "something". Annotation/logging ONLY — it does NOT drive navigation (the
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
