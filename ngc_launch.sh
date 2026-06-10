#!/bin/bash
# NGC Cloud V75 Training Launch Script
# Run this on your NGC instance after cloning the repo

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  OctoTetrahedral AGI V75 - NGC Cloud Training                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check CUDA
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✅ CUDA found: {torch.cuda.get_device_name(0)}')"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
if ! python3 -c "import torch, numpy, yaml" 2>/dev/null; then
    echo "⚠️  Installing missing dependencies..."
    pip install -r requirements.txt
fi
echo "✅ Dependencies OK"
echo ""

# Check data
echo "📂 Checking ARC dataset..."
if [ ! -d "ARC_AMD_TRANSFER/data/ARC-AGI/data" ]; then
    echo "❌ ARC dataset not found!"
    echo "   Please download and place in: ARC_AMD_TRANSFER/data/ARC-AGI/data"
    exit 1
fi
echo "✅ Dataset found"
echo ""

# GPU info
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Launch training
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="train_v75_ngc_${TIMESTAMP}.log"

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 Launching V75 Training"
echo "═══════════════════════════════════════════════════════════════════════"
echo "Configuration:"
echo "  • Steps: 10,000"
echo "  • Batch size: 8"
echo "  • SIMULA: complexity 3, ratio 0.3"
echo "  • Cohesion: ENABLED"
echo "  • Device: CUDA (GPU 0)"
echo "  • Log file: $LOGFILE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

CUDA_VISIBLE_DEVICES=0 nohup python3 train_arc.py \
  --batch-size 8 \
  --max-steps 10000 \
  --use-simula \
  --simula-complexity 3 \
  --simula-ratio 0.3 \
  --use-cohesion \
  > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ Training launched with PID: $PID"
echo "$PID" > train_v75_ngc.pid
echo ""

sleep 5

echo "📊 Initial log output:"
echo "-------------------------------------------------------------------"
tail -30 "$LOGFILE"
echo "-------------------------------------------------------------------"
echo ""
echo "✅ Training running in background"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check progress:"
echo "  grep 'Step' $LOGFILE | tail -20"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🏆 OctoTetrahedral AGI - Mirzakhani's Magic Wand ✨"
echo "═══════════════════════════════════════════════════════════════════════"
