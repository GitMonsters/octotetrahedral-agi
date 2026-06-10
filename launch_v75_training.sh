#!/bin/bash
# V75 Production Training Launch
# Background training with full logging

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="train_v75_${TIMESTAMP}.log"

echo "==================================================================="
echo "OctoTetrahedral AGI - V75 Training Launch"
echo "==================================================================="
echo "Start time: $(date)"
echo "Log file: $LOGFILE"
echo "Configuration:"
echo "  - Steps: 10,000"
echo "  - Batch size: 4"
echo "  - SIMULA: complexity 3, ratio 0.3"
echo "  - Cohesion: DISABLED (MPS memory optimization)"
echo "  - Device: MPS (Apple Silicon)"
echo "==================================================================="

OCTO_DEVICE=mps nohup python3 train_arc.py \
  --batch-size 4 \
  --max-steps 10000 \
  --use-simula \
  --simula-complexity 3 \
  --simula-ratio 0.3 \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "Training launched with PID: $PID"
echo "$PID" > train_v75.pid

sleep 5

echo ""
echo "Initial log output:"
tail -30 "$LOGFILE"

echo ""
echo "==================================================================="
echo "Training running in background (PID: $PID)"
echo "Monitor with: tail -f $LOGFILE"
echo "Stop with: kill $PID"
echo "==================================================================="
