# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.robot – Raspberry Pi 5 Robot Client
# Supports:
#   - Picamera2 + libcamera
#   - GPIO via lgpio/gpiozero
#   - YOLOv8n
#   - OpenCV fallback
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Raspberry Pi Repository
# Needed for python3-libcamera and python3-picamera2
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    && mkdir -p /usr/share/keyrings \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
       | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian bookworm main" \
       > /etc/apt/sources.list.d/raspi.list

# ---------------------------------------------------------------------------
# System packages
#   gcc/g++/python3-dev/libcap-dev : build toolchain for the arm64 pip wheels
#   libatlas-base-dev              : BLAS for numpy/opencv
#   libglib2.0-0                   : required by opencv-python-headless at import
#   libcamera0.5 + python3-libcamera + python3-picamera2 : Pi CSI camera stack
#   python3-kms++ / python3-prctl  : picamera2 runtime helpers
# NOTE: libgl1 is intentionally NOT installed — it is only needed by the GUI
# opencv-python; we use opencv-python-headless (no libGL). libcamera-tools (CLI
# binaries) is also omitted — the app uses the python3-libcamera bindings only.
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libcap-dev \
    libatlas-base-dev \
    libglib2.0-0 \
    libcamera0.5 \
    python3-libcamera \
    python3-picamera2 \
    python3-kms++ \
    python3-prctl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# picamera2 / libcamera are apt packages that install into the *system* Python
# dist-packages (/usr/lib/python3/dist-packages).  The python:3.11-bookworm base
# image runs /usr/local/bin/python3, whose sys.path does NOT include that dir, so
# `import picamera2` fails and the code silently falls back to OpenCV V4L2 — which
# cannot read the Pi CSI camera.
#
# We add dist-packages via a .pth file (site.addsitedir APPENDS it) rather than
# PYTHONPATH (which PREPENDS), so the pip-installed OpenCV in /usr/local stays
# ahead of any apt copy while picamera2 + libcamera remain importable. NOTE:
# path order alone is NOT relied on for numpy — the pip step below force-pins
# numpy 1.26.4 and removes the apt numpy outright (see the ABI comment there).
RUN echo "import site; site.addsitedir('/usr/lib/python3/dist-packages')" \
    > /usr/local/lib/python3.11/site-packages/zzz_dist_packages.pth

WORKDIR /app

# ---------------------------------------------------------------------------
# Copy application
# ---------------------------------------------------------------------------

COPY Code/Robot/ .

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------

# YOLO_ON_PI controls whether YOLOv8n runs on the Pi.
#   1 (default) → install ultralytics/torch; the Pi can run YOLO locally
#                 (DETECTOR_LOCATION=pi, classic split-inference).
#   0           → SKIP ultralytics/torch entirely for a much smaller/faster Pi
#                 image. Use with DETECTOR_LOCATION=server so the PC runs YOLO on
#                 the streamed frames (saves the Pi's CPU/battery). Build with:
#                   docker build --build-arg YOLO_ON_PI=0 -f Dockerfile.robot ...
ARG YOLO_ON_PI=1

# numpy MUST be exactly the 1.26 line: the apt-built picamera2 helper `simplejpeg`
# (in /usr/lib/python3/dist-packages) is compiled against numpy 1.26 (dtype struct
# size 96). An older numpy 1.24 (size 88) or numpy 2.x (size 120) triggers
#   "ValueError: numpy.dtype size changed ... Expected 96 ... got 88/120"
# when picamera2 imports simplejpeg.
#
# Two hazards break the naive pin:
#   1. Bookworm ships an apt numpy 1.24 in dist-packages. Relying only on sys.path
#      order (the .pth append) to keep pip's 1.26 ahead is fragile — on the current
#      apt/pip snapshot the apt 1.24 wins and simplejpeg aborts.
#   2. `pip install ultralytics` can quietly move numpy off 1.26.
# So we (a) pin numpy 1.26.4 LAST with --force-reinstall so nothing downgrades it,
# and (b) delete the stale apt numpy from dist-packages so 1.26.4 is the ONLY numpy
# importable — guaranteeing it matches what simplejpeg was compiled against.
RUN pip install --no-cache-dir \
    "lgpio>=0.2.2.0" \
    "gpiozero>=2.0" \
    "numpy>=1.26,<2" \
    "PyYAML>=6.0.0" \
    "opencv-python-headless>=4.8.0" \
    && if [ "$YOLO_ON_PI" = "1" ]; then \
         pip install --no-cache-dir "ultralytics>=8.0.0"; \
       else \
         echo "YOLO_ON_PI=0 – skipping ultralytics/torch (server-side detection)"; \
       fi \
    && pip install --no-cache-dir --force-reinstall --no-deps "numpy==1.26.4" \
    && rm -rf /usr/lib/python3/dist-packages/numpy \
              /usr/lib/python3/dist-packages/numpy.libs \
              /usr/lib/python3/dist-packages/numpy-*.egg-info \
              /usr/lib/python3/dist-packages/numpy-*.dist-info \
    && python3 -c "import numpy; print('numpy authoritative:', numpy.__version__, numpy.__file__)"

# ---------------------------------------------------------------------------
# Verify the camera stack imports (fail the build if it does not).
# This proves picamera2 + libcamera (apt, in /usr/lib/python3/dist-packages via
# the .pth above) coexist with numpy + OpenCV (pip, in /usr/local site-packages)
# in the SAME interpreter — catching both the "No module named picamera2" fallback
# and any numpy/OpenCV ABI conflict at build time instead of silently at runtime.
# Runs on the Pi (arm64) where this image is built; importing does not need the
# camera hardware (only Picamera2() instantiation does).
# ---------------------------------------------------------------------------

RUN python3 -c "import numpy, cv2, simplejpeg, libcamera, picamera2; from picamera2.encoders import JpegEncoder; print('camera stack import OK: numpy', numpy.__version__, '| cv2', cv2.__version__, '| simplejpeg + picamera2 + libcamera present')"

# ---------------------------------------------------------------------------
# Download YOLO weights during build (only when YOLO runs on the Pi)
# ---------------------------------------------------------------------------

RUN if [ "$YOLO_ON_PI" = "1" ]; then \
      python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true; \
    fi

ENV SERVER_IP=192.168.1.100
# DETECTOR_LOCATION: where YOLO runs in live mode.
#   pi     → Pi runs YOLOv8n locally (default; needs YOLO_ON_PI=1 image)
#   server → Pi streams frames only; the PC runs YOLO (saves Pi battery)
# Override at runtime: docker run -e DETECTOR_LOCATION=server nav-robot
ENV DETECTOR_LOCATION=pi
# GPIO_CHIP: override the gpiochip device number at runtime.
# Pi 5 kernel ≥ 6.6 (pinctrl-rp1): GPIO_CHIP=0  (default)
# Pi 5 kernel < 6.6 or if gpiodetect shows gpiochip4: GPIO_CHIP=4
# Example: docker run -e GPIO_CHIP=4 nav-robot
ENV GPIO_CHIP=0

CMD ["sh", "-c", "python3 main_robot.py --server-ip $SERVER_IP"]