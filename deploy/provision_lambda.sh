#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — Lambda Labs GPU Provisioner
#
# Provisions an H100 GPU instance on Lambda Labs Cloud.
# Cheapest reliable H100: ~$2.49/hr, no egress fees.
#
# Setup:
#   1. Sign up at https://lambda.ai
#   2. Go to https://cloud.lambdalabs.com/api-keys → Generate API key
#   3. Add SSH key at https://cloud.lambdalabs.com/ssh-keys
#
# Usage:
#   export LAMBDA_API_KEY="your-api-key"
#   bash deploy/provision_lambda.sh
# ================================================================

set -euo pipefail

API="https://cloud.lambdalabs.com/api/v1"
LAMBDA_API_KEY="${LAMBDA_API_KEY:-}"
INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_gh200}"
REGION="${REGION:-}"  # auto-select if empty
DOMAIN="${DOMAIN:-api.transcendplexity.ai}"

if [ -z "$LAMBDA_API_KEY" ]; then
    echo "ERROR: No LAMBDA_API_KEY set."
    echo ""
    echo "  1. Sign up at https://lambda.ai"
    echo "  2. Go to https://cloud.lambdalabs.com/api-keys"
    echo "  3. Generate an API key"
    echo "  4. Run: export LAMBDA_API_KEY='your-key'"
    exit 1
fi

AUTH="Authorization: Bearer ${LAMBDA_API_KEY}"
CT="Content-Type: application/json"

echo "============================================"
echo " OctoTetrahedral AGI — Lambda Labs Provisioner"
echo "============================================"
echo ""

# --- 0. Verify auth ---
echo "[0/5] Checking authentication..."
AUTH_CHECK=$(curl -sf -H "$AUTH" "$API/instances" 2>&1) || {
    echo "ERROR: Invalid API key. Check LAMBDA_API_KEY."
    exit 1
}
echo "✅ Authenticated"
echo ""

# --- 1. List available instance types ---
echo "[1/5] Available GPU instances:"
TYPES=$(curl -sf -H "$AUTH" "$API/instance-types")
echo "$TYPES" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
for name, info in sorted(data.items()):
    desc = info.get('instance_type', {})
    price = desc.get('price_cents_per_hour', 0) / 100
    gpu = desc.get('description', 'N/A')
    regions = [r.get('name', '?') for r in info.get('regions_with_capacity_available', [])]
    avail = ', '.join(regions) if regions else 'SOLD OUT'
    print(f'  {name:30s}  \${price:.2f}/hr  [{avail}]')
"
echo ""

# --- 2. Check capacity for requested type ---
echo "[2/5] Checking capacity for ${INSTANCE_TYPE}..."
AVAILABLE_REGION=$(echo "$TYPES" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
target = '${INSTANCE_TYPE}'
preferred = '${REGION}'
info = data.get(target, {})
regions = info.get('regions_with_capacity_available', [])
if preferred:
    for r in regions:
        if r.get('name') == preferred:
            print(r['name']); sys.exit(0)
if regions:
    print(regions[0]['name'])
else:
    print('')
" 2>/dev/null)

if [ -z "$AVAILABLE_REGION" ]; then
    echo "⚠️  ${INSTANCE_TYPE} is sold out everywhere."
    echo "Try a different instance type or check back later."
    echo ""
    echo "Checking alternatives..."
    echo "$TYPES" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
for name, info in sorted(data.items()):
    regions = info.get('regions_with_capacity_available', [])
    if regions:
        desc = info.get('instance_type', {})
        price = desc.get('price_cents_per_hour', 0) / 100
        avail = ', '.join(r['name'] for r in regions)
        print(f'  ✅ {name:35s} \${price:.2f}/hr  [{avail}]')
"
    exit 1
fi
echo "✅ Available in: ${AVAILABLE_REGION}"

# --- 3. Get SSH key ---
echo ""
echo "[3/5] Getting SSH keys..."
SSH_KEYS=$(curl -sf -H "$AUTH" "$API/ssh-keys")
SSH_KEY_ID=$(echo "$SSH_KEYS" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', [])
if data:
    print(data[0]['id'])
else:
    print('')
" 2>/dev/null)

if [ -z "$SSH_KEY_ID" ]; then
    echo "No SSH keys found. Adding yours..."
    if [ -f "$HOME/.ssh/id_ed25519.pub" ]; then
        PUB_KEY=$(cat "$HOME/.ssh/id_ed25519.pub")
    elif [ -f "$HOME/.ssh/id_rsa.pub" ]; then
        PUB_KEY=$(cat "$HOME/.ssh/id_rsa.pub")
    else
        echo "ERROR: No SSH public key found. Run: ssh-keygen -t ed25519"
        exit 1
    fi

    SSH_KEY_NAME="octotetrahedral"
    SSH_RESP=$(curl -sf -H "$AUTH" -H "$CT" -X POST "$API/ssh-keys" \
        -d "{\"name\": \"${SSH_KEY_NAME}\", \"public_key\": \"${PUB_KEY}\"}")
    SSH_KEY_ID=$(echo "$SSH_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
    echo "Added SSH key: ${SSH_KEY_NAME} (${SSH_KEY_ID})"
else
    SSH_KEY_NAME=$(echo "$SSH_KEYS" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['name'])")
    echo "Using SSH key: ${SSH_KEY_NAME} (${SSH_KEY_ID})"
fi

# --- 4. Launch instance ---
echo ""
echo "[4/5] Launching ${INSTANCE_TYPE} in ${AVAILABLE_REGION}..."
LAUNCH_RESP=$(curl -sf -H "$AUTH" -H "$CT" -X POST "$API/instance-operations/launch" \
    -d "{
        \"region_name\": \"${AVAILABLE_REGION}\",
        \"instance_type_name\": \"${INSTANCE_TYPE}\",
        \"ssh_key_names\": [\"${SSH_KEY_NAME}\"],
        \"quantity\": 1,
        \"name\": \"octo-7b-moe\"
    }") || {
    echo "ERROR: Failed to launch instance."
    echo "Response: $LAUNCH_RESP"
    exit 1
}

INSTANCE_IDS=$(echo "$LAUNCH_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin).get('data', {})
ids = data.get('instance_ids', [])
print(','.join(ids))
")
echo "Launched! Instance IDs: ${INSTANCE_IDS}"

# Wait for instance to be ready
echo "Waiting for instance to boot..."
INSTANCE_ID=$(echo "$INSTANCE_IDS" | cut -d',' -f1)
PUBLIC_IP=""
for i in $(seq 1 60); do
    INST=$(curl -sf -H "$AUTH" "$API/instances/${INSTANCE_ID}" 2>/dev/null || true)
    STATUS=$(echo "$INST" | python3 -c "
import sys,json
d = json.load(sys.stdin).get('data',{})
print(d.get('status',''))" 2>/dev/null)
    IP=$(echo "$INST" | python3 -c "
import sys,json
d = json.load(sys.stdin).get('data',{})
print(d.get('ip',''))" 2>/dev/null)

    echo "  status=${STATUS} ip=${IP} (${i}/60)"

    if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then
        PUBLIC_IP="$IP"
        break
    fi
    sleep 10
done

# --- 5. Summary ---
echo ""
echo "============================================"
echo " ✅ Lambda Labs H100 Instance Ready!"
echo "============================================"
echo " Instance:   ${INSTANCE_ID}"
echo " Type:       ${INSTANCE_TYPE}"
echo " Region:     ${AVAILABLE_REGION}"
echo " Public IP:  ${PUBLIC_IP:-pending}"
echo " Domain:     ${DOMAIN}"
echo " Cost:       ~\$2.49/hr (no egress fees)"
echo ""
echo " Next steps:"
echo "   1. SSH in:  ssh ubuntu@${PUBLIC_IP:-<IP>}"
echo "   2. Verify:  nvidia-smi"
echo "   3. Clone:   git clone https://github.com/GitMonsters/octotetrahedral-agi.git"
echo "   4. Install: pip install -r requirements.txt"
echo "   5. Train:   python train_arc_moe.py --config 7b --device cuda:0"
echo ""
echo "   Set DNS: api.transcendplexity.ai → ${PUBLIC_IP:-<IP>}"
echo "   Deploy:  DOMAIN=${DOMAIN} bash deploy/ionos_setup.sh"
echo ""
echo " ⚠️  REMEMBER: Instance costs ~\$2.49/hr while running!"
echo "    To stop: curl -sf -H \"Authorization: Bearer \$LAMBDA_API_KEY\" \\"
echo "      -X POST $API/instance-operations/terminate \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"instance_ids\": [\"${INSTANCE_ID}\"]}'"
echo "============================================"

# Save provisioning info
mkdir -p deploy
cat > deploy/.lambda_provision.json <<EOF
{
    "instance_id": "${INSTANCE_ID}",
    "instance_type": "${INSTANCE_TYPE}",
    "region": "${AVAILABLE_REGION}",
    "public_ip": "${PUBLIC_IP:-}",
    "domain": "${DOMAIN}",
    "ssh_key_id": "${SSH_KEY_ID}"
}
EOF
echo ""
echo "Provisioning info saved to deploy/.lambda_provision.json"
