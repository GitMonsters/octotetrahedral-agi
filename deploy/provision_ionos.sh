#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — IONOS GPU VM Provisioner
#
# Uses the IONOS Cloud REST API (v6) directly via curl.
# Creates an H200 GPU VM in Frankfurt de/fra/2 with a composite call.
#
# Auth: Set IONOS_TOKEN env var (from DCD → Management → Token Manager)
#
# Usage:
#   export IONOS_TOKEN="your-bearer-token"
#   bash deploy/provision_ionos.sh
#
#   # With custom options
#   IONOS_TOKEN=xxx DOMAIN=api.transcendplexity.ai bash deploy/provision_ionos.sh
# ================================================================

set -euo pipefail

API="https://api.ionos.com/cloudapi/v6"
DC_NAME="${DC_NAME:-OctoTetrahedral-GPU}"
DC_LOCATION="${DC_LOCATION:-de/fra}"
SERVER_NAME="${SERVER_NAME:-octo-7b-moe}"
DOMAIN="${DOMAIN:-api.transcendplexity.ai}"
IMAGE_PASSWORD="${IMAGE_PASSWORD:-$(openssl rand -base64 16)}"
IMAGE_ALIAS="${IMAGE_ALIAS:-ubuntu:latest}"

# Try to read token from ionosctl config if not in env
if [ -z "${IONOS_TOKEN:-}" ]; then
    IONOS_TOKEN=$(python3 -c "
import yaml, os
cfg = os.path.expanduser('~/Library/Application Support/ionosctl/config.yaml')
try:
    d = yaml.safe_load(open(cfg))
    for p in d.get('profiles', []):
        t = p.get('credentials', {}).get('token', '')
        if t and t != 'dummy_test':
            print(t); break
except: pass
" 2>/dev/null || true)
fi

if [ -z "${IONOS_TOKEN:-}" ]; then
    echo "ERROR: No IONOS_TOKEN set."
    echo "Generate one at: DCD → Management → Token Manager → Generate"
    echo "Then: export IONOS_TOKEN='your-token'"
    exit 1
fi

AUTH="Authorization: Bearer ${IONOS_TOKEN}"
CT="Content-Type: application/json"

api_get()  { curl -sf -H "$AUTH" -H "$CT" "$API$1"; }
api_post() { curl -sf -H "$AUTH" -H "$CT" -X POST -d "$2" "$API$1"; }

echo "============================================"
echo " OctoTetrahedral AGI — IONOS GPU Provisioner"
echo "============================================"
echo " Domain:  ${DOMAIN}"
echo " Region:  ${DC_LOCATION} (Frankfurt-2, H200)"
echo ""

# --- 0. Verify auth ---
echo "[0/7] Checking authentication..."
USER_INFO=$(curl -sf -H "$AUTH" "https://api.ionos.com/auth/v1/tokens/current" 2>/dev/null) || {
    echo "ERROR: Token invalid or expired."
    echo "Generate a new one at: DCD → Management → Token Manager"
    exit 1
}
echo "✅ Authenticated"
echo ""

# --- 1. List GPU templates ---
echo "[1/7] Finding GPU templates..."
TEMPLATES=$(api_get "/templates?depth=3")
GPU_INFO=$(echo "$TEMPLATES" | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
gpu = [t for t in items if t.get('properties',{}).get('gpus')]
for t in gpu:
    p = t['properties']
    gpus = p.get('gpus', [{}])
    g = gpus[0] if gpus else {}
    print(f\"  {t['id']}  {p['name']:10s}  {p['cores']}c / {p['ram']//1024}GB / {p['storageSize']}GB  GPU: {g.get('count',0)}x {g.get('model','?')}\")
if not gpu:
    # Show all templates if no GPU ones found
    for t in items[:5]:
        p = t['properties']
        print(f\"  {t['id']}  {p['name']:10s}  {p['cores']}c / {p['ram']//1024}GB\")
    print('NO_GPU_TEMPLATES')
") || {
    echo "ERROR: Cannot list templates. GPU access may not be enabled."
    echo "Contact IONOS support to enable Cloud GPU VMs on your contract."
    exit 1
}
echo "$GPU_INFO"

if echo "$GPU_INFO" | grep -q "NO_GPU_TEMPLATES"; then
    echo ""
    echo "⚠️  No GPU templates found. You need to request GPU access first."
    echo "   In DCD, click 'Get your GPU Access enabled now' or contact IONOS support."
    exit 1
fi

# Select best template: prefer H200-S (1 GPU, cheapest)
TEMPLATE_ID=$(echo "$TEMPLATES" | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
gpu = [t for t in items if t.get('properties',{}).get('gpus')]
# Prefer H200-S (1 GPU) for cost efficiency
for t in gpu:
    if 'H200-S' in t['properties'].get('name', ''):
        print(t['id']); sys.exit(0)
# Fallback: smallest GPU template
if gpu:
    gpu.sort(key=lambda t: len(t['properties'].get('gpus', [])))
    print(gpu[0]['id'])
else:
    sys.exit(1)
")
echo ""
echo "Selected: ${TEMPLATE_ID}"

# --- 2. Create data center ---
echo ""
echo "[2/7] Creating data center: ${DC_NAME} in ${DC_LOCATION}..."
EXISTING_DC=$(api_get "/datacenters?depth=1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for dc in data.get('items', []):
    if dc.get('properties', {}).get('name') == '${DC_NAME}':
        print(dc['id']); sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$EXISTING_DC" ]; then
    DC_ID="$EXISTING_DC"
    echo "Data center exists: ${DC_ID}"
else
    DC_RESP=$(api_post "/datacenters" "{
        \"properties\": {
            \"name\": \"${DC_NAME}\",
            \"location\": \"${DC_LOCATION}\",
            \"description\": \"OctoTetrahedral AGI GPU Training\"
        }
    }")
    DC_ID=$(echo "$DC_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created: ${DC_ID}"

    # Wait for DC to be AVAILABLE
    echo "Waiting for data center to provision..."
    for i in $(seq 1 30); do
        STATE=$(api_get "/datacenters/${DC_ID}?depth=0" | python3 -c "
import sys,json; print(json.load(sys.stdin).get('metadata',{}).get('state',''))" 2>/dev/null)
        if [ "$STATE" = "AVAILABLE" ]; then break; fi
        sleep 5
    done
    echo "Data center ready"
fi

# --- 3. Create GPU server (composite call) ---
echo ""
echo "[3/7] Creating GPU server: ${SERVER_NAME} (H200)..."

# Check SSH key
SSH_KEYS_JSON="[]"
for keyfile in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub"; do
    if [ -f "$keyfile" ]; then
        KEY_CONTENT=$(cat "$keyfile")
        SSH_KEYS_JSON="[\"${KEY_CONTENT}\"]"
        echo "Using SSH key: ${keyfile}"
        break
    fi
done

# Check if server already exists
EXISTING_SRV=$(api_get "/datacenters/${DC_ID}/servers?depth=1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('items', []):
    if s.get('properties', {}).get('name') == '${SERVER_NAME}':
        print(s['id']); sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$EXISTING_SRV" ]; then
    SERVER_ID="$EXISTING_SRV"
    echo "Server exists: ${SERVER_ID}"
else
    # Composite call: server + volume in one request (required for GPU VMs)
    SRV_RESP=$(api_post "/datacenters/${DC_ID}/servers" "{
        \"properties\": {
            \"name\": \"${SERVER_NAME}\",
            \"type\": \"GPU\",
            \"availabilityZone\": \"AUTO\",
            \"templateUuid\": \"${TEMPLATE_ID}\"
        },
        \"entities\": {
            \"volumes\": {
                \"items\": [{
                    \"properties\": {
                        \"name\": \"GPU-Boot-Volume\",
                        \"imageAlias\": \"${IMAGE_ALIAS}\",
                        \"imagePassword\": \"${IMAGE_PASSWORD}\",
                        \"sshKeys\": ${SSH_KEYS_JSON}
                    }
                }]
            }
        }
    }")
    SERVER_ID=$(echo "$SRV_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created: ${SERVER_ID}"
    echo "Image password: ${IMAGE_PASSWORD}"

    # Wait for server to provision
    echo "Waiting for GPU server (may take a few minutes)..."
    for i in $(seq 1 60); do
        STATE=$(api_get "/datacenters/${DC_ID}/servers/${SERVER_ID}?depth=0" | python3 -c "
import sys,json; d=json.load(sys.stdin); print(d.get('metadata',{}).get('state',''))" 2>/dev/null)
        VM_STATE=$(api_get "/datacenters/${DC_ID}/servers/${SERVER_ID}?depth=0" | python3 -c "
import sys,json; d=json.load(sys.stdin); print(d.get('properties',{}).get('vmState',''))" 2>/dev/null)
        echo "  state=${STATE} vmState=${VM_STATE} (${i}/60)"
        if [ "$STATE" = "AVAILABLE" ]; then break; fi
        sleep 10
    done
fi

# --- 4. Add LAN + NIC with public IP ---
echo ""
echo "[4/7] Setting up networking..."

# Create a public LAN
LAN_ID=$(api_get "/datacenters/${DC_ID}/lans?depth=1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for lan in data.get('items', []):
    if lan.get('properties', {}).get('public', False):
        print(lan['id']); sys.exit(0)
print('')
" 2>/dev/null)

if [ -z "$LAN_ID" ]; then
    LAN_RESP=$(api_post "/datacenters/${DC_ID}/lans" "{
        \"properties\": {
            \"name\": \"Public-LAN\",
            \"public\": true
        }
    }")
    LAN_ID=$(echo "$LAN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created LAN: ${LAN_ID}"
    sleep 10
else
    echo "Public LAN exists: ${LAN_ID}"
fi

# Create NIC on server attached to the LAN
NIC_ID=$(api_get "/datacenters/${DC_ID}/servers/${SERVER_ID}/nics?depth=1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
if items:
    print(items[0]['id']); sys.exit(0)
print('')
" 2>/dev/null)

if [ -z "$NIC_ID" ]; then
    NIC_RESP=$(api_post "/datacenters/${DC_ID}/servers/${SERVER_ID}/nics" "{
        \"properties\": {
            \"name\": \"Public-NIC\",
            \"lan\": ${LAN_ID},
            \"dhcp\": true,
            \"firewallActive\": true
        }
    }")
    NIC_ID=$(echo "$NIC_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created NIC: ${NIC_ID}"
    sleep 15
else
    echo "NIC exists: ${NIC_ID}"
fi

# --- 5. Firewall rules ---
echo ""
echo "[5/7] Configuring firewall..."
if [ -n "$NIC_ID" ]; then
    for PORT_NAME in "SSH:22:22" "HTTPS:443:443" "HTTP:80:80"; do
        IFS=':' read -r NAME START END <<< "$PORT_NAME"
        api_post "/datacenters/${DC_ID}/servers/${SERVER_ID}/nics/${NIC_ID}/firewallrules" "{
            \"properties\": {
                \"name\": \"${NAME}\",
                \"protocol\": \"TCP\",
                \"portRangeStart\": ${START},
                \"portRangeEnd\": ${END},
                \"direction\": \"INGRESS\"
            }
        }" >/dev/null 2>&1 && echo "  ✅ ${NAME} (${START})" || echo "  ↩️ ${NAME} (may exist)"
    done
fi

# --- 6. Get public IP ---
echo ""
echo "[6/7] Getting public IP..."
sleep 5
PUBLIC_IP=$(api_get "/datacenters/${DC_ID}/servers/${SERVER_ID}/nics?depth=2" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for nic in data.get('items', []):
    ips = nic.get('properties', {}).get('ips', [])
    if ips:
        print(ips[0]); sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$PUBLIC_IP" ]; then
    echo "Public IP: ${PUBLIC_IP}"
else
    echo "⏳ IP not assigned yet. Check DCD or re-run in a few minutes."
fi

# --- 7. Summary ---
echo ""
echo "============================================"
echo " ✅ IONOS H200 GPU VM Provisioned!"
echo "============================================"
echo " Data Center:    ${DC_NAME} (${DC_ID})"
echo " Server:         ${SERVER_NAME} (${SERVER_ID})"
echo " Template:       ${TEMPLATE_ID}"
echo " Public IP:      ${PUBLIC_IP:-pending}"
echo " Domain:         ${DOMAIN}"
echo " Image Password: ${IMAGE_PASSWORD}"
echo ""
echo " Next steps:"
echo "   1. Set DNS A record: api.transcendplexity.ai → ${PUBLIC_IP:-<IP>}"
echo "   2. SSH in:  ssh root@${PUBLIC_IP:-<IP>}"
echo "   3. Verify:  nvidia-smi"
echo "   4. Deploy:  DOMAIN=${DOMAIN} bash deploy/ionos_setup.sh"
echo "   5. Train:   python train_arc_moe.py --config 7b --device cuda:0"
echo "============================================"

# Save provisioning info
mkdir -p deploy
cat > deploy/.ionos_provision.json <<EOF
{
    "datacenter_id": "${DC_ID}",
    "server_id": "${SERVER_ID}",
    "lan_id": "${LAN_ID:-}",
    "nic_id": "${NIC_ID:-}",
    "public_ip": "${PUBLIC_IP:-}",
    "domain": "${DOMAIN}",
    "template_id": "${TEMPLATE_ID}",
    "image_password": "${IMAGE_PASSWORD}"
}
EOF
echo ""
echo "Provisioning info saved to deploy/.ionos_provision.json"
