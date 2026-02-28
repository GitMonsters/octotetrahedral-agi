#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — IONOS GPU VM Setup Script
#
# Provisions an IONOS H100/H200 Cloud GPU VM for model serving.
# Run as root on a fresh Ubuntu 22.04/24.04 IONOS GPU VM.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../ionos_setup.sh | bash
#   # or
#   scp deploy/ionos_setup.sh root@<ionos-ip>:~ && ssh root@<ionos-ip> bash ionos_setup.sh
# ================================================================

set -euo pipefail

MODEL_CONFIG="${MODEL_CONFIG:-7b}"    # 7b | 70b | 1.72t
REPO_URL="${REPO_URL:-https://github.com/GitMonsters/octotetrahedral-agi.git}"
APP_DIR="/opt/octotetrahedral"
PORT=8000

echo "============================================"
echo " OctoTetrahedral AGI — IONOS GPU VM Setup"
echo " Config: ${MODEL_CONFIG}"
echo "============================================"

# --- 1. System packages ---
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git curl docker.io nvidia-container-toolkit 2>/dev/null || true

# Enable NVIDIA container runtime
nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
systemctl restart docker 2>/dev/null || true

# --- 2. Verify GPU ---
echo "[2/6] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "IONOS GPU VMs should have drivers pre-installed."
fi

# --- 3. Clone repo ---
echo "[3/6] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull --quiet
else
    git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# --- 4. Build container ---
echo "[4/6] Building Docker image..."
docker build -f deploy/Dockerfile.gpu -t octotetrahedral:${MODEL_CONFIG}-moe .

# --- 5. Run container ---
echo "[5/6] Starting inference server..."
docker rm -f octotetrahedral 2>/dev/null || true

docker run -d \
    --name octotetrahedral \
    --gpus all \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    -v /opt/octotetrahedral/checkpoints:/app/checkpoints \
    octotetrahedral:${MODEL_CONFIG}-moe \
    python serve.py --config ${MODEL_CONFIG} --device cuda:0

# --- 6. Wait for health ---
echo "[6/6] Waiting for server to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo ""
        echo "============================================"
        echo " ✅ Server is running!"
        echo " API:    http://$(hostname -I | awk '{print $1}'):${PORT}"
        echo " Health: http://$(hostname -I | awk '{print $1}'):${PORT}/health"
        echo " Info:   http://$(hostname -I | awk '{print $1}'):${PORT}/info"
        echo " Docs:   http://$(hostname -I | awk '{print $1}'):${PORT}/docs"
        echo "============================================"
        echo ""
        echo "Test with:"
        echo "  curl -X POST http://localhost:${PORT}/generate \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"prompt\": \"Hello, I am\", \"max_tokens\": 50}'"
        exit 0
    fi
    printf "."
    sleep 5
done

echo ""
echo "WARNING: Server did not become healthy in 150s."
echo "Check logs: docker logs octotetrahedral"
exit 1
