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
DOMAIN="${DOMAIN:-}"               # Set for TLS: DOMAIN=api.example.com
OCTO_API_KEYS="${OCTO_API_KEYS:-}" # Comma-separated API keys

echo "============================================"
echo " OctoTetrahedral AGI — IONOS GPU VM Setup"
echo " Config: ${MODEL_CONFIG}"
echo " Domain: ${DOMAIN:-none (HTTP only)}"
echo "============================================"

# --- 1. System packages ---
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git curl docker.io nvidia-container-toolkit nginx certbot python3-certbot-nginx 2>/dev/null || true

# Enable NVIDIA container runtime
nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
systemctl restart docker 2>/dev/null || true

# --- 2. Verify GPU ---
echo "[2/7] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "IONOS GPU VMs should have drivers pre-installed."
fi

# --- 3. Clone repo ---
echo "[3/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull --quiet
else
    git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

# --- 4. Build container ---
echo "[4/7] Building Docker image..."
docker build -f deploy/Dockerfile.gpu -t octotetrahedral:${MODEL_CONFIG}-moe .

# --- 5. Run container ---
echo "[5/7] Starting inference server..."
docker rm -f octotetrahedral 2>/dev/null || true

DOCKER_ENV=""
if [ -n "$OCTO_API_KEYS" ]; then
    DOCKER_ENV="-e OCTO_API_KEYS=${OCTO_API_KEYS}"
fi

docker run -d \
    --name octotetrahedral \
    --gpus all \
    --restart unless-stopped \
    -p ${PORT}:8000 \
    ${DOCKER_ENV} \
    -v /opt/octotetrahedral/checkpoints:/app/checkpoints \
    octotetrahedral:${MODEL_CONFIG}-moe \
    python serve.py --config ${MODEL_CONFIG} --device cuda:0

# --- 6. Wait for health ---
echo "[6/7] Waiting for server to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo ""
        echo " ✅ Server is running!"
        break
    fi
    printf "."
    sleep 5
done

if ! curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; then
    echo ""
    echo "WARNING: Server did not become healthy in 150s."
    echo "Check logs: docker logs octotetrahedral"
    exit 1
fi

# --- 7. Nginx + TLS ---
echo "[7/7] Configuring reverse proxy..."
if [ -n "$DOMAIN" ]; then
    cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/octotetrahedral
    sed -i "s/YOUR_DOMAIN/${DOMAIN}/g" /etc/nginx/sites-available/octotetrahedral
    ln -sf /etc/nginx/sites-available/octotetrahedral /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    # Temporarily allow HTTP for certbot challenge
    nginx -t && systemctl restart nginx

    # Get TLS cert
    certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos --email "admin@${DOMAIN}" 2>/dev/null || {
        echo "WARNING: certbot failed. Run manually: certbot --nginx -d ${DOMAIN}"
    }

    systemctl reload nginx
    echo ""
    echo "============================================"
    echo " ✅ Deployed with TLS!"
    echo " API:     https://${DOMAIN}/generate"
    echo " Health:  https://${DOMAIN}/health"
    echo " Docs:    https://${DOMAIN}/docs"
    echo " Metrics: http://localhost:${PORT}/metrics (internal only)"
    echo "============================================"
else
    echo " Skipping TLS (no DOMAIN set)."
    IP=$(hostname -I | awk '{print $1}')
    echo ""
    echo "============================================"
    echo " ✅ Deployed (HTTP only)!"
    echo " API:     http://${IP}:${PORT}/generate"
    echo " Health:  http://${IP}:${PORT}/health"
    echo " Docs:    http://${IP}:${PORT}/docs"
    echo " Metrics: http://${IP}:${PORT}/metrics"
    echo "============================================"
    echo ""
    echo " For TLS, re-run with: DOMAIN=api.example.com bash deploy/ionos_setup.sh"
fi

echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:${PORT}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
if [ -n "$OCTO_API_KEYS" ]; then
    FIRST_KEY=$(echo "$OCTO_API_KEYS" | cut -d',' -f1)
    echo "    -H 'X-API-Key: ${FIRST_KEY}' \\"
fi
echo "    -d '{\"prompt\": \"Hello, I am\", \"max_tokens\": 50}'"
