# Troubleshooting — Tank Robot Predictive Navigation

> Part of the [Tank Robot](README.md) docs — see also [ARCHITECTURE.md](ARCHITECTURE.md) · [HOW_TO_RUN.md](HOW_TO_RUN.md) · [CALIBRATION.md](CALIBRATION.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

The hard-won operational gotchas — mostly Pi-side hardware and Docker traps. If a
run misbehaves (no camera frames, motors won't open, flags "do nothing"), check
here first.

---

### Rebuild after pulling changes

The server and robot are **two separate images**. After changing code (or
`git pull`), rebuild **both** (use the same flags as their build commands) or the
stale one keeps its old behaviour:

```bash
docker build --no-cache -f Dockerfile.server -t nav-server .
docker build --no-cache --platform linux/arm64 -f Dockerfile.robot -t nav-robot .
```

`--no-cache` forces a fresh build (skips Docker's layer cache); `--platform
linux/arm64` targets the Pi (drop it if you build on the Pi itself). A robot log
line like `Camera started via picamera2` instead of `Camera streaming via
picamera2 JpegEncoder` means the robot image is stale.

### V-JEPA 2 weights

V-JEPA 2 weights (~300 MB) are downloaded from HuggingFace automatically on first
run. No GPU required — CPU-only inference works out of the box.

### V-JEPA 2 falls back to the stub (log: "V-JEPA 2 load failed … using stub encoder")

If the server logs `V-JEPA 2 load failed (…) – using stub encoder`, the real world
model isn't running and every `wm=` risk you see is synthetic — predictive mode is
effectively baseline. The usual cause is a **transformers version that predates the
`VJEPA2` architecture** (added in **transformers 4.53**). The SSv2 and depth models
load on older builds, so seeing them come up doesn't prove V-JEPA 2 will:

```bash
pip install -U 'transformers>=4.53'      # then restart the server
```

Note: an `Unrecognized processing class … Can't instantiate a processor` message on
its own is harmless — V-JEPA 2 does its own preprocessing and the server now loads
the model without the HF processor. It only falls back to the stub if the **model**
class itself is unavailable (the version issue above).

### Hardware acceleration (MPS / CUDA / Docker-no-Metal)

Both heavy models (V-JEPA 2 and SSv2) default to `device: auto`, which picks the
best available at load: **CUDA → MPS → CPU**. On an NVIDIA host (incl. DGX) run
the GPU image with `--gpus all` (see the GPU build) and they use CUDA; SSv2 also
classifies ~2× more often on a GPU. On a **native** Mac they use Apple **MPS**.
Note that a Docker container **on a Mac** has no Metal passthrough, so it stays on
**CPU** even with unified memory — for GPU on a Mac, run the server natively
(`python main_server.py …`). Force a device by setting `world_model.device` /
`ssv2.device` to `cuda`/`mps`/`cpu`.

### PyQt viewer won't start — "Could not load the Qt platform plugin"

Running `ai_viewer.py` on Linux can abort two ways:

1. **`… plugin "xcb" in ".../cv2/qt/plugins"`** — the *full* `opencv-python` bundles
   its own Qt plugins, which clash with PyQt5. Fix: use the headless build (already
   pinned in `requirements_client.txt`):
   ```bash
   pip uninstall -y opencv-python && pip install opencv-python-headless
   ```
2. **`… plugin "xcb" in ""` even though it was found** — cv2 is out of the way, but
   PyQt5's xcb plugin is missing a system X library. On a **Wayland** desktop the
   quickest fix is to skip xcb entirely:
   ```bash
   QT_QPA_PLATFORM=wayland python3 ai_viewer.py
   # permanent:  echo 'export QT_QPA_PLATFORM=wayland' >> ~/.bashrc
   ```
   To use xcb (via XWayland) instead, install the X libs:
   ```bash
   sudo apt install -y libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
     libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
     libxcb-util1 libxkbcommon-x11-0
   ```
   Pinpoint the exact missing lib with `QT_DEBUG_PLUGINS=1 python3 ai_viewer.py`.

Don't need the desktop window? The browser UI (`http://<PC_IP>:8501`) needs none of this.

### Camera (CSI) — udev mount

The container uses **picamera2/libcamera** to drive the Pi CSI camera (same stack
as Freenove). `-v /run/udev:/run/udev:ro` is **required** so libcamera can
enumerate the camera inside the container; with `--privileged` the camera device
nodes under `/dev` are already available. If picamera2 cannot be imported the code
falls back to OpenCV V4L2 on `/dev/video0`, but the CSI camera generally does
**not** produce frames through that path — so if the PC log says *"waiting for
camera frames"*, confirm the udev mount is present.

### Camera stream freezes after a while (`camera frame is stale`)

The PC logs `camera frame is stale – no NEW frame for 3.0s (stream connected but
not updating)` and the robot appears hung, after streaming fine for a bit. The
robot's `picamera2` hardware encoder **stalled** on a limited buffer pool — the
tell is `picamera2: Failed to open /dev/dma_heap/vidbuf_cached` at startup. Fix:
mount the DMA heap so it uses the proper cached buffers:

```bash
-v /dev/dma_heap:/dev/dma_heap        # add to docker run (compose mounts it already)
```

As a safety net, `camera.py` also runs a watchdog that **auto-restarts** the
backend if no new frame arrives for `camera.stall_timeout_seconds` (default 2.5 s),
and `get_frame()` reports *no frame* during a stall so the robot never drives on a
frozen image (the PC watchdog-STOPs until frames resume). If stalls persist even
with the mount, lower the stream resolution (`camera.stream_width/height`).

### Camera orientation

The feed is streamed upright by default (no flip), so the UI, V-JEPA 2 and YOLO
all get the correctly-oriented frame. If your camera is mounted **inverted** and
the image looks upside-down, flip it with `-e CAMERA_HFLIP=1 -e CAMERA_VFLIP=1`
(no rebuild needed) or set `camera.hflip`/`camera.vflip: true` in
`config_robot.yaml`.

### GPIO chip — both gpiochip mappings

On Pi 5 the RP1 controller is `/dev/gpiochip0`, but the lgpio pin factory also
probes `gpiochip4`, so map the host controller to **both** container nodes
(`--device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip0:/dev/gpiochip4`)
as shown in the run commands. If motors log `can not open gpiochip`, that second
mapping is missing. Run `gpiodetect` on the host to confirm which chip is
`pinctrl-rp1`.

### Motor speed / brownout

The robot drives **slowly** by default (`speed_full: 1600`, `speed_slow: 1000` out
of 4095) so it stays reactive to the CPU pipeline. Tune it **without rebuilding**
via `-e SPEED_FULL=<n> -e SPEED_SLOW=<n>` (or edit `config_robot.yaml`): raise it
a little if the robot doesn't start moving, lower it if it's still too fast. A
soft-start ramp (`robot.soft_start`) blunts the current spike so the Pi doesn't
brown out on drive.

**If the Pi drops off the network the instant it starts to drive** (the camera
stream and `CMD_SONIC` both stop, the PC logs `camera frame is stale` then a cmd
socket timeout, and the robot doesn't reconnect): that's an **inrush brownout /
Wi-Fi glitch** from the motors switching on hard — and it happens *even with a
freshly-charged battery*, because it's about peak current draw (di/dt), not charge.
The soft-start must actually engage: its `soft_start_ramp_step` (default `0.08`)
has to be **below your drive fraction** `speed_full / 4095` or the ramp is skipped
and the motor gets a hard `0 → drive` step. (An earlier default of `0.35` was
*above* a ~0.27 crawl, so soft-start never ran.) If it still browns out, lower
`soft_start_ramp_step` and/or raise `soft_start_ramp_pause` for a gentler ramp,
lower `speed_full`, and check the Pi's 5 V power path and the motor-wire routing
(motor EMI near the Wi-Fi antenna / camera ribbon can also drop the link).

### Undervoltage / power warning (Pi) — a real cause of camera stalls

A Pi **undervoltage warning is not cosmetic** — when the Pi browns out it throttles
the CPU, caps peripheral current, and can **glitch the CSI camera pipeline**, so
frames stop, the PC logs `camera frame is stale`, and the camera watchdog restarts
(and keeps restarting while the power sags). The classic trigger on a robot is the
motors: **motor current spike → the shared battery rail sags → the Pi browns out**,
so the stalls cluster around driving/`REROUTE` bursts, not when it sits still.

A "full" battery can still cause this — voltage-full ≠ able to deliver the current;
the motor inrush drops the rail regardless of charge.

**Confirm it** (on the Pi, ideally while it's driving):

```bash
vcgencmd get_throttled                    # 0x0 = fine; non-zero = a power/thermal event
dmesg | grep -i -E 'voltage|throttl'      # look for "Undervoltage detected!"
watch -n0.5 vcgencmd get_throttled        # watch it flip while the motors move
```

Decode the hex bits: `0x1` under-voltage **now** · `0x4` throttled now ·
`0x10000` under-voltage **has occurred** · `0x40000` throttling has occurred
(e.g. `0x50000` = under-voltage + throttling since boot → power problem confirmed).

**Fix (hardware):** give the **Pi its own regulated 5 V / 5 A** rail, separate from
the motor pack (a dedicated 5 A UBEC/buck off the battery, not shared with the
motors). Pi 5 wants a 5 V/5 A (27 W) source. If you must share the rail, add a large
capacitor (≈1000–4700 µF) across the Pi's 5 V to ride out the inrush, and/or lower
`robot.speed_full`.

This is **separate** from the picamera2 buffer-pool stall (see "Camera stream
freezes" above): the software watchdog restarts the camera either way, but if
`get_throttled` shows undervoltage, **fixing the power supply is the real fix** — the
camera can't stay stable on a sagging rail.

### Compose forwards env vars only if they're declared

With `docker compose`, an inline var like `SPEED_FULL=1500` reaches the container
**only** because `docker-compose.robot.yml` lists it under `environment:`. The
forwarded vars are: `SERVER_IP`, `SPEED_FULL`, `SPEED_SLOW`, `CAMERA_HFLIP`,
`CAMERA_VFLIP`, `GPIO_CHIP`. Example:
`SERVER_IP=192.168.68.107 SPEED_FULL=1500 docker compose -f docker-compose.robot.yml up`.

### `docker compose up` reuses stale images

Both compose files pin an `image:` name, so `up` alone will **not** rebuild after
you change code or `git pull` — it silently reruns the old image and your new
flags appear to "do nothing". Always pass `--build` (or run
`docker compose ... build` first). Build the **robot** image **on the Pi** (arm64
+ Raspberry Pi apt repos); it will not build on a Mac/PC.
