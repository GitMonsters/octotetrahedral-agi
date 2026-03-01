#!/usr/bin/env bash
# ============================================================
#  OctoTetrahedral AGI — Checkpoint Downloader
#  Downloads trained model checkpoints from Lambda GPU instance
#
#  Usage:
#    bash deploy/download_checkpoints.sh                    # all checkpoints
#    bash deploy/download_checkpoints.sh --best-only        # just best model
#    bash deploy/download_checkpoints.sh --ip 1.2.3.4       # custom IP
# ============================================================

set -euo pipefail

# Defaults
REMOTE_USER="ubuntu"
REMOTE_DIR="checkpoints"
LOCAL_DIR="checkpoints/lambda_run1"
BEST_ONLY=false
REMOTE_IP=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --best-only)   BEST_ONLY=true; shift ;;
        --ip)          REMOTE_IP="$2"; shift 2 ;;
        --local-dir)   LOCAL_DIR="$2"; shift 2 ;;
        --remote-dir)  REMOTE_DIR="$2"; shift 2 ;;
        --user)        REMOTE_USER="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect IP from Lambda provision info
if [[ -z "$REMOTE_IP" ]]; then
    PROVISION_FILE="deploy/.lambda_provision.json"
    if [[ -f "$PROVISION_FILE" ]]; then
        REMOTE_IP=$(python3 -c "import json; print(json.load(open('$PROVISION_FILE'))['ip'])" 2>/dev/null || true)
    fi
fi

if [[ -z "$REMOTE_IP" ]]; then
    echo "ERROR: No IP address. Use --ip or run provisioner first."
    exit 1
fi

SSH_TARGET="${REMOTE_USER}@${REMOTE_IP}"

echo "============================================"
echo " OctoTetrahedral — Checkpoint Downloader"
echo "============================================"
echo "Remote: ${SSH_TARGET}:~/${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo ""

# Test SSH
echo "[1/4] Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_TARGET" 'echo ok' >/dev/null 2>&1; then
    echo "ERROR: Cannot SSH to ${SSH_TARGET}"
    exit 1
fi
echo "✅ Connected"

# List remote checkpoints
echo ""
echo "[2/4] Remote checkpoints:"
ssh "$SSH_TARGET" "ls -lhS ~/${REMOTE_DIR}/*.pt 2>/dev/null || echo '  (none found)'"

# Check training status
echo ""
echo "[3/4] Training status:"
TRAIN_PID=$(ssh "$SSH_TARGET" "pgrep -f 'python3 train_gpu.py' 2>/dev/null || true")
if [[ -n "$TRAIN_PID" ]]; then
    echo "  ⚠️  Training is STILL RUNNING (PID: $TRAIN_PID)"
    LAST_LOG=$(ssh "$SSH_TARGET" "tail -1 ~/train.log 2>/dev/null || echo '(no log)'")
    echo "  Last log: $LAST_LOG"
    echo ""
    read -p "  Download available checkpoints anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    echo "  ✅ Training not running (completed or stopped)"
    LAST_LOG=$(ssh "$SSH_TARGET" "tail -3 ~/train.log 2>/dev/null || echo '(no log)'")
    echo "  Last log lines:"
    echo "$LAST_LOG" | sed 's/^/    /'
fi

# Download
echo ""
echo "[4/4] Downloading checkpoints..."
mkdir -p "$LOCAL_DIR"

if $BEST_ONLY; then
    echo "  Downloading arc_best.pt only..."
    scp "${SSH_TARGET}:~/${REMOTE_DIR}/arc_best.pt" "$LOCAL_DIR/" 2>/dev/null && \
        echo "  ✅ arc_best.pt" || echo "  ❌ arc_best.pt not found"
else
    # Download all .pt files
    FILES=$(ssh "$SSH_TARGET" "ls ~/${REMOTE_DIR}/*.pt 2>/dev/null" || true)
    if [[ -z "$FILES" ]]; then
        echo "  No checkpoint files found!"
        exit 1
    fi

    COUNT=0
    for F in $FILES; do
        FNAME=$(basename "$F")
        echo "  Downloading $FNAME..."
        scp "${SSH_TARGET}:${F}" "$LOCAL_DIR/" && COUNT=$((COUNT+1))
    done
    echo "  ✅ Downloaded $COUNT checkpoint(s)"
fi

# Also grab the training log
echo ""
echo "  Downloading train.log..."
scp "${SSH_TARGET}:~/train.log" "$LOCAL_DIR/train.log" 2>/dev/null && \
    echo "  ✅ train.log" || echo "  ⚠️  No train.log found"

# Summary
echo ""
echo "============================================"
echo " Download Complete"
echo "============================================"
echo "Checkpoints saved to: $LOCAL_DIR/"
ls -lhS "$LOCAL_DIR"/*.pt 2>/dev/null || true
echo ""

TOTAL_SIZE=$(du -sh "$LOCAL_DIR" | cut -f1)
echo "Total size: $TOTAL_SIZE"
echo ""
echo "Next steps:"
echo "  # Evaluate on ARC-AGI"
echo "  python eval_arc_moe.py --config 7b --checkpoint $LOCAL_DIR/arc_best.pt --split evaluation"
echo ""
echo "  # Quick eval (10 tasks)"
echo "  python eval_arc_moe.py --config 7b --checkpoint $LOCAL_DIR/arc_best.pt --max-tasks 10"
echo ""
echo "  # TERMINATE the GPU instance (stop billing!)"
echo "  curl -sf -H \"Authorization: Bearer \$LAMBDA_API_KEY\" \\"
echo "    -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"instance_ids\": [\"87cbeb3a051042a089512e0f9ef88568\"]}'"
