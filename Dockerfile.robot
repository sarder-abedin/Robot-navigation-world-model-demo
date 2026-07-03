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
#     --device /dev/gpiochip0:/dev/gpiochip0 \
#     -e SERVER_IP=<PC_IP> \
#     nav-robot
#   Note: Pi 5 kernel ≥ 6.6 uses gpiochip0 (pinctrl-rp1); older Pi 5 kernels
#   used gpiochip4. Run `gpiodetect` on the Pi to confirm the chip label.
#
# Via docker compose (recommended):
#   SERVER_IP=192.168.1.42 docker compose -f docker-compose.robot.yml up
#
# Notes:
#   --privileged   simplest for dev; replace with --device flags for production
#   GPIO chip: Pi 5 kernel ≥ 6.6 → gpiochip0 (pinctrl-rp1)
#              Pi 5 kernel < 6.6 → gpiochip4; Pi 4 → gpiochip0
#   Camera: picamera2 pip pkg is installed; if libcamera is absent the robot
#           falls back to OpenCV VideoCapture via /dev/video0 automatically
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# System libraries available in standard Debian Bookworm.
#
# Camera note: python3-libcamera (the libcamera Python binding) lives in the
# Raspberry Pi Foundation's apt repo, not standard Debian, so it cannot be
# installed here via apt.  camera.py handles this gracefully:
#   • On a Pi with system python3-libcamera → uses picamera2 (best quality)
#   • In Docker with --device /dev/video0    → falls back to OpenCV V4L2
#
# libatlas / libglib / libgl are runtime deps for numpy / OpenCV / gpiozero.
# gcc / python3-dev / libcap-dev compile python-prctl (pulled in by picamera2).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    gcc \
    python3-dev \
    libcap-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy robot code and config
COPY Code/Robot/ .
COPY Code/Server/parameter.py .

# Install Python dependencies including YOLOv8n.
# picamera2 pip wheel installs the Python layer; the required C libcamera bindings
# must be installed on the Pi host (python3-picamera2 via apt) and are accessible
# through the --privileged device passthrough at runtime.
RUN pip install --no-cache-dir \
    "lgpio>=0.2.2.0" \
    "gpiozero>=2.0" \
    "ultralytics>=8.0.0" \
    "picamera2>=0.3.12" \
    "numpy>=1.24.0" \
    "PyYAML>=6.0.0" \
    "opencv-python-headless>=4.8.0"

# Pre-download YOLOv8n weights so the container starts faster
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

ENV SERVER_IP=192.168.1.100

CMD ["sh", "-c", "python main_robot.py --server-ip $SERVER_IP"]
