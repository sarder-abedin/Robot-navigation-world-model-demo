#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — one command to run the WHOLE PC server NATIVELY (not Docker):
#   • sets up a venv + installs deps on first run (auto-picks ROCm / CPU torch),
#   • launches the AI server (YOLO + V-JEPA 2 + SSv2 + depth), and
#   • launches the PyQt operator viewer (world map + V-JEPA 2 foresight),
# then shuts both down cleanly together.
#
# The AI *models* aren't installed by pip — transformers/ultralytics download the
# weights from HuggingFace on the first live run and cache them in
# ~/.cache/huggingface (and ./yolo11n.pt). Nothing extra to do.
#
# Usage:
#   ./start.sh                 # setup on first run, then live mode + viewer
#   ./start.sh --setup         # force re-install deps into the venv
#   ./start.sh --demo          # demo mode (needs assets/demo_clips/corridor.mp4)
#   ./start.sh --nav baseline  # reactive-only baseline
#   ./start.sh --no-ui         # server only (no PyQt viewer)
#   ./start.sh --cpu           # setup the CPU/CUDA deps even if a Radeon is present
#
# Env: NAV_MODE, NAV_STRATEGY, ROCM_INDEX (default rocm6.3 wheel index).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/roboenv"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}"

MODE="${NAV_MODE:-live}"
NAV="${NAV_STRATEGY:-predictive}"
DO_SETUP=0
LAUNCH_UI=1
FORCE_CPU=0

while [ $# -gt 0 ]; do
  case "$1" in
    --setup)  DO_SETUP=1 ;;
    --demo)   MODE="demo" ;;
    --mode)   MODE="${2:?--mode needs a value}"; shift ;;
    --nav)    NAV="${2:?--nav needs a value}"; shift ;;
    --no-ui)  LAUNCH_UI=0 ;;
    --cpu)    FORCE_CPU=1 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "start.sh: unknown argument '$1' (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ── 1. First-run setup: venv + the right torch + deps ────────────────────────
if [ ! -x "$VENV/bin/python" ] || [ "$DO_SETUP" = 1 ]; then
  echo "[start] Setting up the virtualenv at $VENV …"
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip
  if [ "$FORCE_CPU" = 0 ] && [ -e /dev/kfd ]; then
    echo "[start] AMD GPU detected (/dev/kfd) → ROCm torch from $ROCM_INDEX, then requirements-rocm.txt"
    pip install --index-url "$ROCM_INDEX" torch torchvision
    pip install -r "$REPO/requirements-rocm.txt"
  else
    echo "[start] Installing default deps (CPU / CUDA torch from PyPI) → requirements_server.txt"
    pip install -r "$REPO/requirements_server.txt"
  fi
  echo "[start] Setup complete."
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# ── 2. Launch the AI server + PyQt viewer, shut down together ────────────────
SERVER_PID=""
UI_PID=""
cleanup() {
  echo "[start] Stopping…"
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$UI_PID" ]     && kill "$UI_PID"     2>/dev/null || true
}
trap cleanup EXIT INT TERM

# All AI runs inside main_server.py. --no-display: the annotated video shows in the
# viewer, so we don't also need the server's own OpenCV HUD window. Logs → logs_rpi/.
cd "$REPO/Code/Server"
python main_server.py --mode "$MODE" --nav "$NAV" --no-display &
SERVER_PID=$!
echo "[start] AI server started (PID $SERVER_PID) — mode=$MODE nav=$NAV. Loading models…"

if [ "$LAUNCH_UI" = 1 ]; then
  sleep 2   # let the server bind its ports before the viewer connects
  # On a Wayland session, use the Wayland Qt platform so PyQt doesn't abort on xcb.
  if [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
  fi
  cd "$REPO/Code/Client"
  python ai_viewer.py &
  UI_PID=$!
  echo "[start] PyQt viewer started (PID $UI_PID) — leave Server IP as 127.0.0.1 and click Connect."
else
  echo "[start] Viewer skipped (--no-ui). Open the browser UI or run Code/Client/ai_viewer.py yourself."
fi

# Exit as soon as EITHER process exits (e.g. the UI's Shutdown button kills the
# server), then the trap stops whichever is still alive — one clean shutdown.
wait -n
echo "[start] A process exited — shutting everything down."
