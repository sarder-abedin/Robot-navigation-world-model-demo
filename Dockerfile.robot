# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.robot – Raspberry Pi 5 Robot Client (thin client, no on-Pi AI)
#
# All AI (YOLOv8n + V-JEPA 2 + SSv2) runs on the PC server. This image is
# deliberately lightweight: it carries NO torch / ultralytics. It provides:
#   - Picamera2 + libcamera (camera streaming)
#   - GPIO via lgpio/gpiozero (motors)
#   - Ultrasonic sensor (local hard-stop safety)
#   - OpenCV (JPEG encode/decode) + numpy
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

# Lightweight dependency set — NO torch/ultralytics (all AI runs on the PC).
#
# numpy MUST be exactly the 1.26 line: the apt-built picamera2 helper `simplejpeg`
# (in /usr/lib/python3/dist-packages) is compiled against numpy 1.26 (dtype struct
# size 96). An older numpy 1.24 (size 88) or numpy 2.x (size 120) triggers
#   "ValueError: numpy.dtype size changed ... Expected 96 ... got 88/120"
# when picamera2 imports simplejpeg. Bookworm ships an apt numpy 1.24 in
# dist-packages, and relying on sys.path order (the .pth append) to keep pip's
# 1.26 ahead is fragile. So we (a) pin numpy 1.26.4 LAST with --force-reinstall so
# nothing downgrades it, and (b) delete the stale apt numpy from dist-packages so
# 1.26.4 is the ONLY numpy importable — matching what simplejpeg was compiled with.
RUN pip install --no-cache-dir \
    "lgpio>=0.2.2.0" \
    "gpiozero>=2.0" \
    "numpy>=1.26,<2" \
    "PyYAML>=6.0.0" \
    "opencv-python-headless>=4.8.0" \
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

ENV SERVER_IP=192.168.1.100
# GPIO_CHIP: override the gpiochip device number at runtime.
# Pi 5 kernel ≥ 6.6 (pinctrl-rp1): GPIO_CHIP=0  (default)
# Pi 5 kernel < 6.6 or if gpiodetect shows gpiochip4: GPIO_CHIP=4
# Example: docker run -e GPIO_CHIP=4 nav-robot
ENV GPIO_CHIP=0

CMD ["sh", "-c", "python3 main_robot.py --server-ip $SERVER_IP"]