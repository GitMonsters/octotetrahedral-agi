#!/bin/bash
# ================================================================
# OctoTetrahedral AGI -- One-Command Deploy
#
# Provisions a GPU, sets it up, trains, and starts serving.
# Works with RunPod, Lambda Labs, or any existing SSH box.
#
# Usage:
#   # RunPod (cheapest H100)
#   export RUNPOD_API_KEY="your-key"
#   bash deploy/deploy.sh runpod
#
#   # Lambda Labs
#   export LAMBDA_API_KEY="your-key"
#   bash deploy/deploy.sh lambda
#
#   # Existing SSH box
#   bash deploy/deploy.sh ssh root@209.20.159.131
#
# Options (env vars):
#   SCALE=tiny     Model scale preset
#   TRAIN=1        Run training
#   SERVE=1        Start API server
#   EPOCHS=3       Training epochs
#   NUM_TASKS=400  ARC tasks to train on
# ================================================================

set -euo pipefail

SCALE="${SCALE:-tiny}"
TRAIN="${TRAIN:-1}"
SERVE="${SERVE:-1}"
EPOCHS="${EPOCHS:-3}"
NUM_TASKS="${NUM_TASKS:-400}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PROVIDER="${1:-}"

if [ -z "$PROVIDER" ]; then
    echo "Usage: bash deploy/deploy.sh <provider>"
    echo ""
    echo "Providers:"
    echo "  runpod              Provision RunPod H100 (~\$1.99/hr)"
    echo "  lambda              Provision Lambda Labs GPU (~\$2.49/hr)"
    echo "  ssh user@host       Deploy to existing SSH box"
    echo ""
    echo "Environment variables:"
    echo "  SCALE=tiny          Model scale (tiny|base|large|ultra)"
    echo "  TRAIN=1             Run ARC training"
    echo "  SERVE=1             Start API server"
    echo "  EPOCHS=3            Training epochs"
    echo "  NUM_TASKS=400       Number of ARC tasks"
    exit 1
fi

echo "============================================"
echo " OctoTetrahedral AGI -- Deploy"
echo " Provider: ${PROVIDER}"
echo " Scale: ${SCALE} | Train: ${TRAIN} | Serve: ${SERVE}"
echo "============================================"
echo ""

get_ssh_target() {
    local provider="$1"
    case "$provider" in
        runpod)
            echo "[1/2] Provisioning RunPod..."
            bash "${SCRIPT_DIR}/provision_runpod.sh"
            if [ -f "${SCRIPT_DIR}/.runpod_provision.json" ]; then
                local ip port
                ip=$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/.runpod_provision.json'))['public_ip'])")
                port=$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/.runpod_provision.json'))['ssh_port'])")
                echo "root@${ip} -p ${port}"
            else
                echo "ERROR: RunPod provisioning failed" >&2
                exit 1
            fi
            ;;
        lambda)
            echo "[1/2] Provisioning Lambda Labs..."
            bash "${SCRIPT_DIR}/provision_lambda.sh"
            if [ -f "${SCRIPT_DIR}/.lambda_provision.json" ]; then
                local ip
                ip=$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/.lambda_provision.json'))['public_ip'])")
                echo "ubuntu@${ip}"
            else
                echo "ERROR: Lambda provisioning failed" >&2
                exit 1
            fi
            ;;
        ssh)
            shift 2>/dev/null || true
            echo "${@:2}"
            ;;
        *)
            # Assume it's a direct ssh target like user@host
            echo "$provider"
            ;;
    esac
}

SSH_TARGET=$(get_ssh_target "$PROVIDER" "$@")
echo ""
echo "[2/2] Deploying to: ${SSH_TARGET}"
echo ""

# Upload and run remote setup
# shellcheck disable=SC2086
scp -o StrictHostKeyChecking=no ${SSH_TARGET##* -p *} "${SCRIPT_DIR}/remote_setup.sh" "${SSH_TARGET%% *}":~/setup.sh 2>/dev/null \
    || scp -o StrictHostKeyChecking=no "${SCRIPT_DIR}/remote_setup.sh" ${SSH_TARGET%%:*}:~/setup.sh 2>/dev/null \
    || { echo "Trying simple scp..."; scp -o StrictHostKeyChecking=no "${SCRIPT_DIR}/remote_setup.sh" "$(echo $SSH_TARGET | awk '{print $1}'):~/setup.sh"; }

# Run the setup remotely
# shellcheck disable=SC2086
ssh -o StrictHostKeyChecking=no $SSH_TARGET \
    "TRAIN=${TRAIN} SERVE=${SERVE} SCALE=${SCALE} EPOCHS=${EPOCHS} NUM_TASKS=${NUM_TASKS} bash ~/setup.sh"

echo ""
echo "============================================"
echo " Deployment complete!"
echo " SSH: ssh ${SSH_TARGET}"
echo "============================================"
