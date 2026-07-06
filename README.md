# Tank Robot – Predictive Indoor Navigation

A predictive indoor navigation system for the **Freenove Tank Robot Kit for
Raspberry Pi (FNK0077)** that uses **V-JEPA 2** as a world model to anticipate
future obstacles — not just react to what is currently visible.

---

## What

- Predictive indoor navigation for the **Freenove FNK0077** tank robot.
- **Split-inference** architecture: the Pi is a thin client that only streams
  camera frames + ultrasonic distance; **all AI runs on the PC** — YOLO11n +
  V-JEPA 2 + SSv2 + Depth-Anything V2.
- **Multi-layer safety**, each layer only ever making the robot *more* cautious:
  ultrasonic hard-stop → vision reroute → kinematic speed governor → depth
  free-space.
- Operator UI viewers (browser Streamlit, native pywebview window, PyQt5) show
  live annotated video, risk bars and AI state; AUTO (AI drives) / MANUAL modes.

## Why

- **V-JEPA 2 anticipates** obstacles instead of only reacting — it begins
  decelerating frames before a YOLO-only baseline would.
- **Depth gives class-agnostic geometry** — it sees walls YOLO has no class for,
  and yields a real REROUTE direction (which side is open).
- **SSv2 reads behaviour** — a real Something-Something-V2 video model classifies
  what the obstacle is *doing* and fills in YOLO's object (e.g. *"person moving
  closer"*), giving the operator and logs a human-readable read of the scene
  (annotation only — it does not drive control).
- **The latency-aware governor** reserves braking room for the AI's decision
  time, so the robot never outruns its own perception (e.g. slow V-JEPA 2 on CPU).
- **The thin-client Pi stays lightweight** — no torch/ultralytics on the robot.

## How

Fastest path — demo in Docker on Mac / Linux (no robot needed):

```bash
# Build the server image (arm64 Mac Apple Silicon + amd64 Linux both work)
docker build -f Dockerfile.server -t nav-server .

# Demo mode — supply a corridor video first:
#   mkdir -p assets/demo_clips && cp /path/to/corridor.mp4 assets/demo_clips/
docker run --rm -p 5003:5003 -p 8003:8003 -p 5004:5004 -p 8004:8004 -p 8501:8501 -v "$(pwd)/assets:/app/assets:ro" -v "$(pwd)/logs_rpi:/app/logs_rpi" nav-server
# Then open http://localhost:8501 → enter "localhost" as server IP → Connect
```

For live mode, the native venv path, GPU builds and the full Docker flags, see
**[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

## First-run checklist

1. **Install/build** — server deps in a venv on the PC (`pip install -r requirements_server.txt`); build the robot image on the Pi. → [HOW_TO_RUN.md](HOW_TO_RUN.md)
2. **Smoke-test in demo mode** (no robot): run the server `--mode demo`, open the UI, confirm annotated video + risk/action HUD. → [HOW_TO_RUN.md](HOW_TO_RUN.md)
3. **Go live**: start the server (natively on a Mac for MPS), start the Pi client (`docker compose -f docker-compose.robot.yml up --build`), confirm the UI shows live video and a *changing* ultrasonic value. → [HOW_TO_RUN.md](HOW_TO_RUN.md)
4. **Calibrate V-JEPA 2 anchors**: capture ~10 "blocked" + ~10 "clear" corridor frames, run `python Code/Server/calibrate_anchors.py --blocked ./blocked --clear ./clear --out anchors.npz`, then set `world_model.anchors_path: "anchors.npz"` in `Code/Server/config.yaml`. → [HOW_TO_RUN.md](HOW_TO_RUN.md)
5. **Calibrate the speed governor**: on the robot, facing a flat wall with clear runway, run `python Code/Robot/calibrate_governor.py --apply ../Server/config.yaml` (measures forward/slow speed + max decel, patches config safely). → [HOW_TO_RUN.md](HOW_TO_RUN.md)
6. **Verify**: the HUD shows a sane `Depth: <d>m ahead | open: <dir>` line and the log shows `SCENE: <obj> <d>m ahead | open=<dir>`; the robot slows early, stops before walls, and reroutes toward the open side.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — design, AI models, TCP protocol, safety layers, project structure.
- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** — setup, run modes (demo/live/Docker), UI viewers, anchor + governor calibration, config tables, tests, logging.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — camera / GPIO / compose / brownout gotchas and the rebuild-after-pull trap.
- `CLAUDE.md` — development guide for AI assistants working in this repo.

## Success criteria

- Robot reaches the goal point reliably in both modes
- Predictive mode visibly begins decelerating **earlier** than baseline mode
- The V-JEPA 2 label shows `BLOCKED` before the YOLO11 detector fills the risk bar
- Motion is smoother in predictive mode (fewer full stops from a cold start)
- System runs stably at ≥ 8 FPS in demo mode
- All signals are logged to CSV for post-run analysis
