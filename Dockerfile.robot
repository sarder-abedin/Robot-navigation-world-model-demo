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
# cannot read the Pi CSI camera.  Add dist-packages to the path so picamera2 works.
ENV PYTHONPATH=/usr/lib/python3/dist-packages

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