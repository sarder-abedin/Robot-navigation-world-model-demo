# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.robot – Raspberry Pi robot client (arm64 / linux/arm64)
#
# Hardware-only image: camera streaming, motor control, ultrasonic sensor.
# No AI models. Connects outbound to the PC AI server.
#
# Build (on Pi or via cross-compilation):
#   docker build --platform linux/arm64 -f Dockerfile.robot -t nav-robot .
#
# Run (on Raspberry Pi):
#   docker run --privileged \
#     --device /dev/video0:/dev/video0 \
#     --device /dev/gpiochip4:/dev/gpiochip4 \
#     -e SERVER_IP=<PC_IP> \
#     nav-robot
#
# Notes:
#   --privileged   simplest for dev; replace with specific --device flags for prod
#   /dev/gpiochip4 is the GPIO chip on Pi 5 (Pi 4 uses /dev/gpiochip0)
#   libcamera system libs must be installed on the host; picamera2 is pip-installed
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# System libraries for libcamera / picamera2
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

RUN pip install --no-cache-dir \
    lgpio>=0.2.2.0 \
    gpiozero>=2.0 \
    numpy>=1.24.0 \
    PyYAML>=6.0.0 \
    opencv-python-headless>=4.8.0

ENV SERVER_IP=192.168.1.100

CMD ["sh", "-c", "python main_robot.py --server-ip $SERVER_IP"]
