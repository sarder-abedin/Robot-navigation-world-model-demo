# Calibration Guide

> Part of the [Tank Robot](README.md) docs — see also [ARCHITECTURE.md](ARCHITECTURE.md) · [HOW_TO_RUN.md](HOW_TO_RUN.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

The navigation stack runs **without** calibration — but several signals are only
*qualitatively* right until you calibrate them for your robot and space. This
guide covers all three calibrations in detail.

## What needs calibrating, and why

| Calibration | Fixes | Symptom when skipped | Effort |
|---|---|---|---|
| **1. V-JEPA 2 anchors** | The world-model risk + `BLOCKED`/`CLEAR` label | Label stuck on `MIXED`, `world_model_risk` pinned ~0.48, no predictive early-warning | ~5 min, once |
| **2. Speed governor** | The kinematic safe-speed cap (SI m/s constants) | Robot crawls/stops everywhere or doesn't brake in time; `d_stop` math is guesswork | ~10 min, on the robot |
| **3. Depth scale** | The metric distance from Depth-Anything | Wrong "X.XX m ahead", wrong goal-arrival point, governor mis-caps | ~5 min, once |

**Calibration-free by design (no action needed):** the **reroute turn direction**
uses *relative* per-side depth (which side is more open), and the **ultrasonic
hard-stop** is distance-only from the sonar. Those work uncalibrated. What needs
calibration is anything using an *absolute* number: the world-model risk score,
the governor's metres, and the goal's arrival distance.

**Do it in this order:** anchors (biggest quality win) → depth scale (feeds the
governor + goal arrival) → governor (depends on depth + motor reality).

---

## ⚡ Zero-driving calibration from logs (recommended)

You don't need to drive *for* calibration. A normal run already records everything
needed, and the numbers are derived offline — the ultrasonic distance logged each
frame is a free ground-truth ruler.

**Easiest: the calibration UI** (a separate desk-only PyQt5 window — it never drives
the robot):
```bash
python Code/Server/calibration_ui.py
```
It's a **step-by-step guided workflow**: each numbered step shows a badge
(**MANDATORY** / **RECOMMENDED** / **OPTIONAL**) and a live status (done / do this
next / waiting / not-ready) so you always know what to do next. The guidance is a
*soft guide* — it points you at the recommended next step and flags steps that
aren't ready, but never blocks you from running a step out of order. The steps:

| Step | Level | What it does |
|---|---|---|
| **0 · Record a run** | Mandatory | Prerequisite — a normal run logged with a working ultrasonic (and `save_raw_frames` on for anchors). Info only; nothing here drives the robot. |
| **1 · Select run(s)** | Mandatory | Tick one run (✓CSV required, ✓raw needed for anchors). **Pooling several runs is optional** but more robust. **Scan logs dir** lists runs; **+ Add run folder…** pulls one in from anywhere. |
| **2 · Analyze** | Mandatory | Derive the values without writing anything; fills in the status of Steps 3–5. |
| **3 · Depth scale** | Recommended | `median(sonar ÷ depth)` readout — needs enough valid sonar/depth pairs. |
| **4 · Governor speeds** | Recommended | forward/slow m/s + deceleration from distance-vs-time. |
| **5 · V-JEPA 2 anchors** | Recommended | Tick “Build anchors on Apply” — needs a ✓raw run (auto-disabled otherwise). |
| **6 · Apply to config** | Mandatory | Write the derived values into `config.yaml` (comments kept, `.bak` backup) and verify what landed. |
| **7 · Restart the server** | Mandatory | The server reads `config.yaml` only at startup. |

Or use the CLI **`calibrate_from_logs.py --run <dir> [more…]`** directly: same
math, same results.

**One-time setup** — turn on raw-frame capture (only needed for anchors):
```yaml
# Code/Server/config.yaml
logging:
  save_raw_frames: true     # writes logs_rpi/<run>/raw_frames/ (un-annotated)
```
Then **do one normal run** (Obstacle Avoidance is fine) with **run logging on** and
a **working ultrasonic**, letting the robot approach and stop near obstacles a few
times. That single run's folder is your calibration data.

**Then, at your desk** — pass **one or more** run folders (more runs = more robust;
they're pooled):
```bash
cd Code/Server
# depth scale + governor speeds (numpy only), pooled across several runs:
python calibrate_from_logs.py --run ../../logs_rpi/run_A ../../logs_rpi/run_B
# review the printed values, then write them in (also builds anchors if raw frames exist):
python calibrate_from_logs.py --run ../../logs_rpi/run_* --anchors --apply config.yaml
```
`--apply` patches `config.yaml`, then **re-reads it and prints the effective values +
the absolute config path**, and reminds you to **restart the server** (it reads
`config.yaml` at startup) so the calibrated values take effect on the next run. The
anchors path is written **absolute** so the server finds it regardless of its
working directory.

What it computes:
- **`depth.scale`** = median(ultrasonic ÷ depth) over frames with a valid sonar reading.
- **`forward_speed_mps` / `slow_speed_mps`** = distance-vs-time slope during FORWARD/SLOW
  stretches; **`max_decel_mps2`** from a FORWARD→STOP coast when present (else the
  config default is kept).
- **anchors** (`--anchors`) — auto-labels the raw frames blocked/clear from *independent*
  signals (YOLO + ultrasonic + action, **never** the world model) and builds anchors
  with the same encoder as the manual tool. No sorting.

**Requirements & limits:**
- The **ultrasonic must have been working** in that run (frames with no echo are
  skipped; the tool says "insufficient" if too few remain). This is the one thing to
  confirm first.
- **Anchors** only work on runs recorded *after* `save_raw_frames` was turned on
  (older runs have annotated frames only). Run `--anchors` where V-JEPA 2 loads.
- Governor **deceleration** is the least reliable from logs (short coast on a tank
  robot); when it can't measure one it leaves your configured default in place.

The manual, in-corridor procedures below are the fallback when you can't get a clean
logged run (e.g. the sonar is dead).

---

## 1. Calibrate the V-JEPA 2 anchors

**What it does.** V-JEPA 2 scores each scene by cosine similarity to two reference
embeddings — a **blocked** anchor and a **clear** anchor:

```
diff = similarity(scene, blocked) − similarity(scene, clear)
BLOCKED if diff > world_model.label_margin · CLEAR if diff < −margin · else MIXED
```

Out of the box those anchors are **synthetic** (a grey square vs a gradient), so a
real corridor sits almost exactly between them → `diff ≈ 0` → the label is always
`MIXED` and the risk barely moves. Calibrating replaces them with embeddings of
**your** environment, so the score actually discriminates blocked vs clear.

### Step by step

1. **Collect frames** — two folders of images (JPEG/PNG) captured on your robot in
   your space:
   - `blocked/` — the path blocked (a wall / person / obstacle filling the view).
   - `clear/` — the path open (nothing in the way).

   ~10–20 images per class, varied (different spots, lighting, obstacle types).
   You can snapshot from the robot, save frames from the run log
   (`logs_rpi/<run>/frames/`), or grab frames from the demo video.

2. **Build the anchors** — run **where V-JEPA 2 actually loads** (the native
   GPU/MPS PC, *not* a CPU-only Docker container), so the anchors come from the
   real encoder and not the stub:

   ```bash
   cd Code/Server
   python calibrate_anchors.py --blocked ./blocked --clear ./clear --out anchors.npz
   ```

   Flags: `--blocked` / `--clear` (required folders), `--out` (default
   `anchors.npz`), `--config` (default `config.yaml`).

3. **Point the server at them** — in `Code/Server/config.yaml`:

   ```yaml
   world_model:
     anchors_path: "anchors.npz"
   ```

4. **Verify** — restart the server; it logs `V-JEPA 2 anchors loaded from … (calibrated)`
   on startup when they're in use. Then watch the HUD/panel: `V-JEPA 2` should now
   read `BLOCKED` in front of an obstacle and `CLEAR` down an open corridor, and
   `world_model_risk` in the run log should swing (not sit at ~0.48).

### Tips & troubleshooting
- **Still mostly MIXED?** Your two classes are too similar — make `blocked/` clearly
  blocked and `clear/` clearly open. You can also lower `world_model.label_margin`
  (default `0.02`) so smaller differences flip the label.
- **Ran on CPU/stub?** The tool warns if V-JEPA 2 didn't load; the resulting anchors
  are worthless. Run it on the box where the model loads.
- Re-run any time your environment changes materially (new space, very different
  lighting).

---

## 2. Calibrate the depth scale

**What it does.** Depth-Anything V2 (metric model) outputs distance in metres, but
the absolute scale can be off for your camera/mount. Depth feeds three absolute-
distance consumers: the **HUD distance**, the **speed governor** (`clear_distance_m`),
and the **goal arrival** check. A single linear correction fixes the scale.

### Step by step

1. **Measure ground truth.** Put a flat obstacle (wall/box) a **tape-measured**
   distance straight ahead — e.g. exactly **1.00 m** from the camera.
2. **Read what depth reports.** Enable the depth-distance HUD text (set
   `visualization.overlay_depth_text: true` in `config.yaml`) or read the `Depth:`
   line in the UI panel / the `clear_dist_m` column in the run log. Note the value,
   e.g. it says **1.35 m**.
3. **Compute the scale** = `actual / reported` = `1.00 / 1.35 ≈ 0.74`.
4. **Set it** in `Code/Server/config.yaml`:

   ```yaml
   depth:
     scale: 0.74     # 1.0 = raw model metres
   ```

5. **Verify** at a *different* distance (say 2.0 m) — the corrected reading should
   now match within ~10%. Repeat at 2–3 distances and average the scale if they
   disagree; if it's non-linear at close range, calibrate around the distances you
   care about (obstacle/arrival range).

### Notes
- The correction is **linear** (`depth × scale`). If your model is badly non-linear,
  prefer a different metric checkpoint via `depth.model_id`.
- Depth calibration directly sets where **"Goal reached"** triggers
  (`goal.arrival_distance_m`) and how the governor caps speed — do it before
  trusting either.
- The reroute *direction* does **not** need this (it uses relative depth), so an
  uncalibrated scale won't stop the robot rerouting — only its absolute distances
  will be wrong.

---

## 3. Calibrate the speed governor

**What it does.** The governor caps the action so the robot can always stop within
the clear distance ahead, given its reaction latency:

```
d_stop(v) = v·t_react + (v² − v_target²) / (2·a) + margin
```

The math is in **SI units** (m, s, m/s, m/s²) but the robot drives in PWM, so the
constants (`forward_speed_mps`, `slow_speed_mps`, `max_decel_mps2`) must be
**measured on your robot**. The defaults are conservative guesses, not truth.

### Option A — the on-robot script (recommended)

Run **on the robot** (needs motor + ultrasonic), pointed at a **flat wall with
~2–3 m of clear runway**:

```bash
cd Code/Robot
python calibrate_governor.py                          # measure + print the block
python calibrate_governor.py --apply ../Server/config.yaml   # measure + patch safely
```

It drives at the FORWARD/SLOW PWM and fits **distance-vs-time from the sonar** for
the speeds, then commands STOP and uses the **coast distance** for the deceleration
(`a = v² / 2d`). It takes the **median of several runs**, **aborts** if the wall
gets within `--min-distance-cm` (default 30), and **refuses to emit/apply
physically implausible values**.

Useful flags: `--config` (robot config for the PWM values), `--apply <server config>`
(patch in place), `--runs` (default 3), `--drive-seconds` (default 1.5),
`--min-distance-cm` (default 30).

`--apply` is **safe**: it sanity-checks → backs up the config → patches only the
governor numerics (comments preserved) → re-validates the YAML → restores the
backup on any failure. The script runs on the Pi but the governor lives in the
**PC's** `config.yaml`, so either `--apply` a synced copy or paste the printed
block on the PC.

> Set the same PWM you run with — the script reads `SPEED_FULL`/`SPEED_SLOW`
> (env or `config_robot.yaml`). If you change robot speed later, re-calibrate.

### Option B — measure by hand

Edit `Code/Server/config.yaml → decision.governor`:

| Constant | How to measure |
|---|---|
| `forward_speed_mps` | Drive at the FORWARD PWM over a tape-measured distance; speed = distance ÷ time. |
| `slow_speed_mps` | Same at the SLOW PWM. |
| `max_decel_mps2` | From `forward_speed_mps`, command STOP and measure the coast distance `d`; `a ≈ v² / (2d)`. |
| `target_speed_mps` | `0` for a full stop, or a small crawl speed to only *reduce* impact. |
| `safety_margin_m` | Fixed buffer (e.g. 0.10 m) for sensor/timing slop. |
| `min_reaction_s` / `max_reaction_s` | Floor/cap on the pipeline's measured reaction latency (see `reaction_ema_ms` in the run log). |

Set `decision.governor.enabled: false` to turn the governor off entirely and fall
back to the fixed risk thresholds + ultrasonic hard-stop.

### Verify
Drive toward a wall: the robot should **downgrade FORWARD→SLOW→STOP** and come to
rest with `safety_margin_m` to spare — not brake too late, and not crawl when the
path is genuinely open. If it's over-cautious everywhere, your `*_speed_mps` are
likely too high or depth scale is under-reading distance (calibrate depth first).

---

## Post-calibration checklist

- [ ] Server logs `V-JEPA 2 anchors loaded from … (calibrated)` at startup.
- [ ] HUD `V-JEPA 2` shows `BLOCKED`/`CLEAR` (not stuck on `MIXED`), and
      `world_model_risk` in the log varies.
- [ ] Depth HUD distance matches a tape measure within ~10% at your working range.
- [ ] Governor visibly steps FORWARD→SLOW→STOP approaching a wall and stops with margin.
- [ ] "Goal reached" fires at roughly the real `goal.arrival_distance_m`.

## Config quick-reference

```yaml
world_model:
  anchors_path: "anchors.npz"   # from calibrate_anchors.py
  label_margin: 0.02            # lower → flips BLOCKED/CLEAR on smaller diffs
depth:
  scale: 1.0                    # actual / reported distance
  model_id: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
goal:
  arrival_distance_m: 0.4       # goal depth ≤ this → "reached"
decision:
  governor:
    forward_speed_mps: 0.35     # measured — calibrate_governor.py
    slow_speed_mps: 0.18        # measured
    max_decel_mps2: 0.6         # measured
    safety_margin_m: 0.10
    enabled: true               # false = governor off
```
