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
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libcap-dev \
    libatlas-base-dev \
    libglib2.0-0 \
    libgl1 \
    libcamera0.5 \
    libcamera-tools \
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
# PYTHONPATH (which PREPENDS).  Appending keeps the pip-installed numpy/OpenCV in
# /usr/local ahead of the apt numpy in dist-packages, avoiding a numpy ABI clash
# that would otherwise break `import cv2`, while still making picamera2 + libcamera
# importable.
RUN echo "import site; site.addsitedir('/usr/lib/python3/dist-packages')" \
    > /usr/local/lib/python3.11/site-packages/zzz_dist_packages.pth

WORKDIR /app

# ---------------------------------------------------------------------------
# Copy application
# ---------------------------------------------------------------------------

COPY Code/Robot/ .
COPY Code/Server/parameter.py .

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------

RUN pip install --no-cache-dir \
    "lgpio>=0.2.2.0" \
    "gpiozero>=2.0" \
    "ultralytics>=8.0.0" \
    "numpy>=1.24.0" \
    "PyYAML>=6.0.0" \
    "opencv-python-headless>=4.8.0"

# ---------------------------------------------------------------------------
# Verify the camera stack imports (fail the build if it does not).
# This proves picamera2 + libcamera (apt, in /usr/lib/python3/dist-packages via
# the .pth above) coexist with numpy + OpenCV (pip, in /usr/local site-packages)
# in the SAME interpreter — catching both the "No module named picamera2" fallback
# and any numpy/OpenCV ABI conflict at build time instead of silently at runtime.
# Runs on the Pi (arm64) where this image is built; importing does not need the
# camera hardware (only Picamera2() instantiation does).
# ---------------------------------------------------------------------------

RUN python3 -c "import numpy, cv2, picamera2, libcamera; print('camera stack import OK: numpy', numpy.__version__, '| cv2', cv2.__version__, '| picamera2 + libcamera present')"

# ---------------------------------------------------------------------------
# Download YOLO weights during build
# ---------------------------------------------------------------------------

RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

ENV SERVER_IP=192.168.1.100
# GPIO_CHIP: override the gpiochip device number at runtime.
# Pi 5 kernel ≥ 6.6 (pinctrl-rp1): GPIO_CHIP=0  (default)
# Pi 5 kernel < 6.6 or if gpiodetect shows gpiochip4: GPIO_CHIP=4
# Example: docker run -e GPIO_CHIP=4 nav-robot
ENV GPIO_CHIP=0

CMD ["sh", "-c", "python3 main_robot.py --server-ip $SERVER_IP"]