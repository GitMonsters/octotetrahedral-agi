#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — RunPod GPU Provisioner
#
# Provisions an H100 GPU pod on RunPod.
# Competitive pricing: ~$1.99-2.39/hr for H100.
#
# Setup:
#   1. Sign up at https://runpod.io
#   2. Go to Settings → API Keys → Generate
#   3. Add SSH key in Settings → SSH Keys
#
# Usage:
#   export RUNPOD_API_KEY="your-api-key"
#   bash deploy/provision_runpod.sh
# ================================================================

set -euo pipefail

API="https://api.runpod.io/graphql"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
GPU_TYPE="${GPU_TYPE:-NVIDIA H100 80GB HBM3}"
DOMAIN="${DOMAIN:-api.transcendplexity.ai}"
DISK_SIZE="${DISK_SIZE:-100}"     # GB persistent disk
VOLUME_SIZE="${VOLUME_SIZE:-200}" # GB network volume

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: No RUNPOD_API_KEY set."
    echo ""
    echo "  1. Sign up at https://runpod.io"
    echo "  2. Settings → API Keys → Create API Key"
    echo "  3. Run: export RUNPOD_API_KEY='your-key'"
    exit 1
fi

gql() {
    local query="$1"
    curl -sf -X POST "$API?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\"}"
}

echo "============================================"
echo " OctoTetrahedral AGI — RunPod Provisioner"
echo "============================================"
echo ""

# --- 0. Verify auth ---
echo "[0/4] Checking authentication..."
AUTH_CHECK=$(gql "{ myself { id email } }") || {
    echo "ERROR: Invalid API key."
    exit 1
}
EMAIL=$(echo "$AUTH_CHECK" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['myself']['email'])" 2>/dev/null || echo "unknown")
echo "✅ Authenticated as: ${EMAIL}"
echo ""

# --- 1. Check GPU availability ---
echo "[1/4] Checking GPU availability..."
GPU_QUERY="{ gpuTypes { id displayName memoryInGb secureCloud communityCloud } }"
GPU_DATA=$(gql "$GPU_QUERY")
echo "$GPU_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {}).get('gpuTypes', [])
for g in data:
    if 'h100' in g.get('displayName', '').lower() or 'a100' in g.get('displayName', '').lower():
        secure = '✅' if g.get('secureCloud') else '❌'
        community = '✅' if g.get('communityCloud') else '❌'
        mem = g.get('memoryInGb', '?')
        print(f\"  {g['displayName']:35s} {mem}GB  Secure:{secure}  Community:{community}  ID:{g['id']}\")
"

GPU_ID=$(echo "$GPU_DATA" | python3 -c "
import sys, json
target = '${GPU_TYPE}'
data = json.load(sys.stdin).get('data', {}).get('gpuTypes', [])
for g in data:
    if g.get('displayName', '') == target:
        print(g['id']); sys.exit(0)
# fallback: any H100
for g in data:
    if 'h100' in g.get('displayName', '').lower():
        print(g['id']); sys.exit(0)
print('')
" 2>/dev/null)

if [ -z "$GPU_ID" ]; then
    echo "⚠️  No H100 available. Try A100 or check back."
    exit 1
fi
echo "Using GPU: ${GPU_ID}"
echo ""

# --- 2. Create pod ---
echo "[2/4] Creating pod with ${GPU_TYPE}..."
CREATE_QUERY="mutation { podFindAndDeployOnDemand( input: { cloudType: ALL, gpuCount: 1, volumeInGb: ${VOLUME_SIZE}, containerDiskInGb: ${DISK_SIZE}, gpuTypeId: \\\"${GPU_ID}\\\", name: \\\"octo-7b-moe\\\", imageName: \\\"pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel\\\", dockerArgs: \\\"\\\", ports: \\\"22/tcp,8000/http,8080/http\\\", volumeMountPath: \\\"/workspace\\\" } ) { id desiredStatus imageName machine { podExternalId } } }"
CREATE_RESP=$(gql "$CREATE_QUERY") || {
    echo "ERROR: Failed to create pod."
    exit 1
}

POD_ID=$(echo "$CREATE_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'errors' in data:
    print('ERROR: ' + str(data['errors']), file=sys.stderr)
    sys.exit(1)
pod = data.get('data', {}).get('podFindAndDeployOnDemand', {})
print(pod.get('id', ''))
" 2>/dev/null)

if [ -z "$POD_ID" ] || [ "$POD_ID" = "None" ]; then
    echo "ERROR: Pod creation failed."
    echo "$CREATE_RESP" | python3 -m json.tool 2>/dev/null || echo "$CREATE_RESP"
    exit 1
fi
echo "✅ Pod created: ${POD_ID}"

# --- 3. Wait for pod to be ready ---
echo ""
echo "[3/4] Waiting for pod to be ready..."
POD_IP=""
SSH_PORT=""
for i in $(seq 1 60); do
    POD_QUERY="{ pod(input: { podId: \\\"${POD_ID}\\\" }) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } gpus { id gpuUtilPercent memoryUtilPercent } } } }"
    POD_STATUS=$(gql "$POD_QUERY")
    STATUS=$(echo "$POD_STATUS" | python3 -c "
import sys, json
d = json.load(sys.stdin).get('data', {}).get('pod', {})
status = d.get('desiredStatus', 'unknown')
runtime = d.get('runtime')
if runtime:
    ports = runtime.get('ports', [])
    for p in (ports or []):
        if p.get('privatePort') == 22 and p.get('isIpPublic'):
            print(f\"READY|{p.get('ip', '')}|{p.get('publicPort', '')}\")
            sys.exit(0)
print(f'{status}|pending|pending')
" 2>/dev/null)

    STATE=$(echo "$STATUS" | cut -d'|' -f1)
    IP=$(echo "$STATUS" | cut -d'|' -f2)
    PORT=$(echo "$STATUS" | cut -d'|' -f3)

    echo "  status=${STATE} ip=${IP} port=${PORT} (${i}/60)"

    if [ "$STATE" = "READY" ]; then
        POD_IP="$IP"
        SSH_PORT="$PORT"
        break
    fi
    sleep 10
done

# --- 4. Summary ---
echo ""
echo "============================================"
echo " ✅ RunPod H100 Instance Ready!"
echo "============================================"
echo " Pod ID:     ${POD_ID}"
echo " GPU:        ${GPU_TYPE}"
echo " Public IP:  ${POD_IP:-pending}"
echo " SSH Port:   ${SSH_PORT:-pending}"
echo " Domain:     ${DOMAIN}"
echo " Cost:       ~\$1.99-2.39/hr"
echo ""
echo " Next steps:"
echo "   1. SSH in:  ssh root@${POD_IP:-<IP>} -p ${SSH_PORT:-<PORT>} -i ~/.ssh/id_ed25519"
echo "   2. Verify:  nvidia-smi"
echo "   3. Clone:   git clone https://github.com/GitMonsters/octotetrahedral-agi.git /workspace/octo"
echo "   4. Install: cd /workspace/octo && pip install -r requirements.txt"
echo "   5. Train:   python train_arc_moe.py --config 7b --device cuda:0"
echo ""
echo " ⚠️  REMEMBER: Pod costs money while running!"
echo "    To stop: curl -sf -X POST '$API?api_key=\$RUNPOD_API_KEY' \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\": \"mutation { podStop(input: { podId: \\\\\"${POD_ID}\\\\\" }) { id desiredStatus } }\"}'"
echo "============================================"

# Save provisioning info
mkdir -p deploy
cat > deploy/.runpod_provision.json <<EOF
{
    "pod_id": "${POD_ID}",
    "gpu_type": "${GPU_TYPE}",
    "public_ip": "${POD_IP:-}",
    "ssh_port": "${SSH_PORT:-}",
    "domain": "${DOMAIN}"
}
EOF
echo ""
echo "Provisioning info saved to deploy/.runpod_provision.json"
