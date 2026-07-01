# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.robot – Raspberry Pi robot client (arm64 / linux/arm64)
#
# Split-inference responsibilities on the Pi:
#   • Runs YOLOv8n locally for fast reactive obstacle detection
#   • Streams JPEG camera frames to PC (port 8004) for V-JEPA 2 inference
#   • Sends CMD_DETECTION (detection + ultrasonic) to PC (port 5004)
#   • Receives CMD_AIMOVE (AI actions) and CMD_MOTOR (manual) from PC
#   • Executes motor PWM via tankMotor (gpiozero)
#
# Build (on Pi or via cross-compilation):
#   docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .
#
# Run (on Raspberry Pi – replace <PC_IP> with your laptop/PC IP):
#   docker run --privileged \
#     --device /dev/video0:/dev/video0 \
#     --device /dev/gpiochip4:/dev/gpiochip4 \
#     -e SERVER_IP=<PC_IP> \
#     nav-robot
#
# Via docker compose (recommended):
#   SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up
#
# Notes:
#   --privileged   simplest for dev; replace with --device flags for production
#   /dev/gpiochip4 is the GPIO chip on Pi 5 (Pi 4 uses /dev/gpiochip0)
#   libcamera system libs need to be installed on the host; picamera2 from apt
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# System libraries for libcamera / picamera2 and OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcamera-dev \
    libcamera-apps-lite \
    python3-libcamera \
    python3-kms++ \
    python3-prctl \
    python3-picamera2 \
    libatlas-base-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy robot code and config
COPY Code/Robot/ .
COPY Code/Server/parameter.py .

# Install Python dependencies including YOLOv8n
RUN pip install --no-cache-dir \
    "lgpio>=0.2.2.0" \
    "gpiozero>=2.0" \
    "ultralytics>=8.0.0" \
    "numpy>=1.24.0" \
    "PyYAML>=6.0.0" \
    "opencv-python-headless>=4.8.0"

# Pre-download YOLOv8n weights so the container starts faster
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

ENV SERVER_IP=192.168.1.100

CMD ["sh", "-c", "python main_robot.py --server-ip $SERVER_IP"]
