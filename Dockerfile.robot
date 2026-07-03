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

CMD ["sh", "-c", "python3 main_robot.py --server-ip $SERVER_IP"]