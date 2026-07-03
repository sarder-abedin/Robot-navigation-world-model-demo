#!/bin/bash
# start_server.sh – Launch the AI navigation server and Streamlit UI viewer together.
# Used as the default CMD in Dockerfile.server.
#
# Environment variables:
#   NAV_MODE       demo | live   (default: demo)
#   NAV_STRATEGY   predictive | baseline   (default: predictive)

SERVER_PID=""

cleanup() {
    echo "[start_server] Stopping..."
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

# ── 1. Start the AI navigation server in background ──────────────────────────
cd /app/Code/Server
python main_server.py \
    --mode "${NAV_MODE:-demo}" \
    --nav  "${NAV_STRATEGY:-predictive}" \
    --no-display &
SERVER_PID=$!
echo "[start_server] AI server started (PID $SERVER_PID)"

# ── 2. Give the server 2 seconds to bind its ports ───────────────────────────
sleep 2

# ── 3. Start Streamlit viewer in foreground ───────────────────────────────────
# When Streamlit exits the container exits, triggering cleanup → kills server.
echo "[start_server] Starting Streamlit on http://0.0.0.0:8501"
streamlit run /app/Code/Client/streamlit_viewer.py \
    --server.port           8501 \
    --server.address        0.0.0.0 \
    --server.headless       true \
    --browser.gatherUsageStats false
