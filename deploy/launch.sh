#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — One-Command Cloud Launch
#
# Provisions IONOS GPU VM + deploys multi-modal model in one shot.
# Requires: IONOS_TOKEN environment variable
#
# Usage:
#   export IONOS_TOKEN="your-bearer-token"
#   bash deploy/launch.sh
#
#   # With options:
#   IONOS_TOKEN=xxx SCALE=base DOMAIN=api.transcendplexity.ai bash deploy/launch.sh
#
# Scale presets:
#   tiny  → 204M  params (default, 1 GPU)
#   base  → 26.2B params (needs 8 GPUs)
#   large → 1.66T params (needs 128 GPUs)
#   ultra → 1.75T params (needs 512+ GPUs, distributed)
# ================================================================

set -euo pipefail

SCALE="${SCALE:-tiny}"
DOMAIN="${DOMAIN:-api.transcendplexity.ai}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI — Cloud Launch              ║"
echo "║  Scale: ${SCALE} | Domain: ${DOMAIN}"
echo "║  Genus-13 Multi-Modal (text+vision+audio+embod)  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Step 1: Provision IONOS GPU VM
echo "=== Step 1/3: Provisioning IONOS GPU VM ==="
DOMAIN="${DOMAIN}" bash "${SCRIPT_DIR}/provision_ionos.sh"

# Read the provisioned IP
if [ -f "${SCRIPT_DIR}/.ionos_provision.json" ]; then
    PUBLIC_IP=$(python3 -c "import json; d=json.load(open('${SCRIPT_DIR}/.ionos_provision.json')); print(d.get('public_ip',''))")
else
    echo "ERROR: Provisioning info not found. Check provision_ionos.sh output."
    exit 1
fi

if [ -z "$PUBLIC_IP" ]; then
    echo "ERROR: No public IP assigned. Wait a few minutes and check IONOS DCD."
    exit 1
fi

echo ""
echo "=== Step 2/3: Setting up GPU VM (${PUBLIC_IP}) ==="
# Upload setup script and run it
scp -o StrictHostKeyChecking=no "${SCRIPT_DIR}/ionos_setup.sh" root@${PUBLIC_IP}:~/setup.sh
ssh -o StrictHostKeyChecking=no root@${PUBLIC_IP} \
    "MODEL_CONFIG=${SCALE} DOMAIN=${DOMAIN} bash ~/setup.sh"

echo ""
echo "=== Step 3/3: Verifying deployment ==="
sleep 10

# Test health endpoint
if curl -sf "http://${PUBLIC_IP}:8000/health" >/dev/null 2>&1; then
    echo "✅ Health check passed!"
elif [ -n "$DOMAIN" ]; then
    # Try via domain
    if curl -sf "https://${DOMAIN}/health" >/dev/null 2>&1; then
        echo "✅ Health check passed (via ${DOMAIN})!"
    else
        echo "⚠️  Health check pending. Server may still be starting up."
    fi
else
    echo "⚠️  Health check pending. Server may still be starting up."
fi

# Get model info
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  🚀 DEPLOYMENT COMPLETE                         ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  IP:      ${PUBLIC_IP}"
echo "║  Domain:  ${DOMAIN}"
echo "║  Scale:   ${SCALE}"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Endpoints:                                      ║"
echo "║    Health:  https://${DOMAIN}/health"
echo "║    Info:    https://${DOMAIN}/info"
echo "║    Generate: https://${DOMAIN}/generate"
echo "║    Stream:  https://${DOMAIN}/generate/stream"
echo "║    Metrics: http://${PUBLIC_IP}:8000/metrics"
echo "║    Docs:    https://${DOMAIN}/docs"
echo "╠══════════════════════════════════════════════════╣"
echo "║  SSH:  ssh root@${PUBLIC_IP}"
echo "║  GPU:  ssh root@${PUBLIC_IP} nvidia-smi"
echo "║  Logs: ssh root@${PUBLIC_IP} docker logs -f octotetrahedral"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Test:"
echo "  curl -X POST https://${DOMAIN}/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Hello, I am\", \"max_tokens\": 50}'"
