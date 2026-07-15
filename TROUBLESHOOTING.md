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

### Camera (CSI) — udev mount

The container uses **picamera2/libcamera** to drive the Pi CSI camera (same stack
as Freenove). `-v /run/udev:/run/udev:ro` is **required** so libcamera can
enumerate the camera inside the container; with `--privileged` the camera device
nodes under `/dev` are already available. If picamera2 cannot be imported the code
falls back to OpenCV V4L2 on `/dev/video0`, but the CSI camera generally does
**not** produce frames through that path — so if the PC log says *"waiting for
camera frames"*, confirm the udev mount is present.

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
