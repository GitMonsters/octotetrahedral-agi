#!/bin/bash
# ================================================================
# OctoTetrahedral AGI — IONOS GPU VM Provisioner
#
# Provisions a GPU VM, sets up firewall, deploys the model.
# Requires: ionosctl authenticated (run `ionosctl login` first)
#
# Usage:
#   # Step 1: Authenticate
#   ionosctl login
#
#   # Step 2: Provision and deploy
#   bash deploy/provision_ionos.sh
#
#   # With custom options
#   DOMAIN=api.transcendplexity.ai OCTO_API_KEYS=mysecretkey bash deploy/provision_ionos.sh
# ================================================================

set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

# Configuration
DC_NAME="${DC_NAME:-OctoTetrahedral-GPU}"
DC_LOCATION="${DC_LOCATION:-de/fra}"       # Frankfurt
SERVER_NAME="${SERVER_NAME:-octo-7b-moe}"
DOMAIN="${DOMAIN:-api.transcendplexity.ai}"
OCTO_API_KEYS="${OCTO_API_KEYS:-}"
MODEL_CONFIG="${MODEL_CONFIG:-7b}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519.pub}"

echo "============================================"
echo " OctoTetrahedral AGI — IONOS GPU Provisioner"
echo "============================================"
echo " Domain:  ${DOMAIN}"
echo " Config:  ${MODEL_CONFIG}"
echo " Region:  ${DC_LOCATION}"
echo ""

# --- 0. Verify auth ---
echo "[0/8] Checking authentication..."
ionosctl whoami -o json 2>/dev/null | head -5 || {
    echo "ERROR: Not authenticated. Run: ionosctl login"
    exit 1
}
echo ""

# --- 1. List available GPU templates ---
echo "[1/8] Finding GPU templates..."
ionosctl template list --cols TemplateId,Name,Cores,RAM,StorageSize,GPUs -o text 2>/dev/null || {
    echo "ERROR: Cannot list templates. GPU access may not be enabled on your contract."
    echo "Contact IONOS support to enable Cloud GPU VMs."
    exit 1
}
echo ""

# Auto-select first GPU template (usually H100)
TEMPLATE_ID=$(ionosctl template list -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
# Prefer templates with GPUs > 0
gpu_templates = [t for t in items if t.get('properties', {}).get('gpus', 0) > 0]
if gpu_templates:
    # Prefer smallest GPU template for cost
    gpu_templates.sort(key=lambda t: t.get('properties', {}).get('gpus', 0))
    print(t['id'] for t in gpu_templates)
    print(gpu_templates[0]['id'])
elif items:
    print(items[0]['id'])
else:
    sys.exit(1)
" 2>/dev/null | tail -1) || {
    echo "No GPU templates found. Enter template ID manually:"
    read -r TEMPLATE_ID
}
echo "Selected template: ${TEMPLATE_ID}"

# --- 2. Create data center ---
echo ""
echo "[2/8] Creating data center: ${DC_NAME} in ${DC_LOCATION}..."
DC_ID=$(ionosctl datacenter list -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for dc in data.get('items', []):
    if dc.get('properties', {}).get('name') == '${DC_NAME}':
        print(dc['id'])
        sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$DC_ID" ]; then
    echo "Data center already exists: ${DC_ID}"
else
    DC_ID=$(ionosctl datacenter create \
        --name "${DC_NAME}" \
        --location "${DC_LOCATION}" \
        --wait-for-request \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created data center: ${DC_ID}"
fi

# --- 3. Create GPU server ---
echo ""
echo "[3/8] Creating GPU server: ${SERVER_NAME}..."

# Check if SSH key exists
SSH_ARGS=""
if [ -f "${SSH_KEY_PATH}" ]; then
    SSH_ARGS="--ssh-key-paths ${SSH_KEY_PATH}"
    echo "Using SSH key: ${SSH_KEY_PATH}"
elif [ -f "$HOME/.ssh/id_rsa.pub" ]; then
    SSH_ARGS="--ssh-key-paths $HOME/.ssh/id_rsa.pub"
    echo "Using SSH key: $HOME/.ssh/id_rsa.pub"
else
    echo "WARNING: No SSH key found. Set a password or generate one: ssh-keygen -t ed25519"
fi

# Check if server already exists
SERVER_ID=$(ionosctl server list --datacenter-id "${DC_ID}" -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('items', []):
    if s.get('properties', {}).get('name') == '${SERVER_NAME}':
        print(s['id'])
        sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$SERVER_ID" ]; then
    echo "Server already exists: ${SERVER_ID}"
else
    SERVER_ID=$(ionosctl server create \
        --datacenter-id "${DC_ID}" \
        --name "${SERVER_NAME}" \
        --type GPU \
        --template-id "${TEMPLATE_ID}" \
        --licence-type LINUX \
        ${SSH_ARGS} \
        --wait-for-request \
        --wait-for-state \
        -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "Created GPU server: ${SERVER_ID}"
fi

# --- 4. Get server details ---
echo ""
echo "[4/8] Getting server details..."
ionosctl server get --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" \
    --cols ServerId,Name,Type,Cores,RAM,VmState -o text

# List GPUs
echo ""
echo "Attached GPUs:"
ionosctl server gpu list --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" -o text 2>/dev/null || echo "(none visible yet)"

# --- 5. Create firewall rules ---
echo ""
echo "[5/8] Configuring firewall..."

# Get the NIC ID
NIC_ID=$(ionosctl nic list --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
if items:
    print(items[0]['id'])
" 2>/dev/null)

if [ -n "$NIC_ID" ]; then
    # Allow SSH (port 22)
    ionosctl firewallrule create \
        --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" --nic-id "${NIC_ID}" \
        --name "SSH" --protocol TCP --port-range-start 22 --port-range-end 22 \
        --direction INGRESS --wait-for-request 2>/dev/null || echo "SSH rule may already exist"

    # Allow HTTPS (port 443)
    ionosctl firewallrule create \
        --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" --nic-id "${NIC_ID}" \
        --name "HTTPS" --protocol TCP --port-range-start 443 --port-range-end 443 \
        --direction INGRESS --wait-for-request 2>/dev/null || echo "HTTPS rule may already exist"

    echo "Firewall configured: SSH (22) + HTTPS (443)"
else
    echo "WARNING: No NIC found. Configure firewall manually in IONOS DCD."
fi

# --- 6. Get public IP ---
echo ""
echo "[6/8] Getting public IP..."
PUBLIC_IP=$(ionosctl nic list --datacenter-id "${DC_ID}" --server-id "${SERVER_ID}" -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
items = data.get('items', [])
for nic in items:
    ips = nic.get('properties', {}).get('ips', [])
    if ips:
        print(ips[0])
        sys.exit(0)
print('')
" 2>/dev/null)

if [ -n "$PUBLIC_IP" ]; then
    echo "Public IP: ${PUBLIC_IP}"
else
    echo "WARNING: No public IP assigned yet. Check IONOS DCD."
    echo "You may need to add a LAN + NIC with a public IP."
fi

# --- 7. DNS instructions ---
echo ""
echo "[7/8] DNS Setup..."
if [ -n "$PUBLIC_IP" ] && [ -n "$DOMAIN" ]; then
    echo "Point ${DOMAIN} to ${PUBLIC_IP}"
    echo ""
    echo "In IONOS Domain settings for transcendplexity.ai:"
    echo "  Type: A"
    echo "  Host: api"
    echo "  Points to: ${PUBLIC_IP}"
    echo "  TTL: 3600"
    echo ""
    echo "After DNS propagates, deploy with:"
    echo "  ssh root@${PUBLIC_IP} 'DOMAIN=${DOMAIN} OCTO_API_KEYS=${OCTO_API_KEYS:-changeme} MODEL_CONFIG=${MODEL_CONFIG} bash <(curl -sSL https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/deploy/ionos_setup.sh)'"
fi

# --- 8. Summary ---
echo ""
echo "============================================"
echo " ✅ IONOS GPU VM Provisioned!"
echo "============================================"
echo " Data Center: ${DC_NAME} (${DC_ID})"
echo " Server:      ${SERVER_NAME} (${SERVER_ID})"
echo " Public IP:   ${PUBLIC_IP:-pending}"
echo " Domain:      ${DOMAIN}"
echo ""
echo " Next steps:"
echo "   1. Set DNS: api.transcendplexity.ai → ${PUBLIC_IP:-<IP>}"
echo "   2. SSH in:  ssh root@${PUBLIC_IP:-<IP>}"
echo "   3. Deploy:  DOMAIN=${DOMAIN} OCTO_API_KEYS=<key> bash deploy/ionos_setup.sh"
echo "   4. Train:   python train_arc_moe.py --config 7b --device cuda:0"
echo "   5. Eval:    python eval_arc_moe.py --config 7b --checkpoint checkpoints/best.pt"
echo "============================================"

# Save provisioning info
cat > deploy/.ionos_provision.json <<EOF
{
    "datacenter_id": "${DC_ID}",
    "server_id": "${SERVER_ID}",
    "public_ip": "${PUBLIC_IP}",
    "domain": "${DOMAIN}",
    "template_id": "${TEMPLATE_ID}",
    "model_config": "${MODEL_CONFIG}"
}
EOF
echo ""
echo "Provisioning info saved to deploy/.ionos_provision.json"
