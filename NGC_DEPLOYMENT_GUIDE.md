# NGC Cloud Deployment Guide - V75 Training

## Overview
Deploy OctoTetrahedral AGI V75 training on NVIDIA NGC Cloud for fast CUDA-accelerated training.

## Prerequisites
- NGC Cloud account (cloud.nvidia.com)
- GitHub repo access: GitMonsters/octotetrahedral-agi
- ~20 GB VRAM GPU (RTX 4090, A100, or H100)

## Step 1: Launch NGC Instance

1. Go to https://cloud.nvidia.com
2. Launch a new instance:
   - **Framework**: PyTorch 2.x
   - **GPU**: A100 (40GB) or H100 (recommended)
   - **Storage**: 100 GB minimum
   - **Instance type**: Single GPU is sufficient

## Step 2: Clone Repository

```bash
# SSH into NGC instance, then:
git clone https://github.com/GitMonsters/octotetrahedral-agi.git
cd octotetrahedral-agi
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Verify CUDA

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA A100-SXM4-40GB (or similar)
```

## Step 5: Launch V75 Training

```bash
# Full training with all features
CUDA_VISIBLE_DEVICES=0 nohup python3 train_arc.py \
  --batch-size 8 \
  --max-steps 10000 \
  --use-simula \
  --simula-complexity 3 \
  --simula-ratio 0.3 \
  --use-cohesion \
  > train_v75_ngc.log 2>&1 &

# Save PID
echo $! > train_v75_ngc.pid
```

## Step 6: Monitor Training

```bash
# Watch live progress
tail -f train_v75_ngc.log

# Check GPU utilization
nvidia-smi -l 1

# Check specific training steps
grep "Step" train_v75_ngc.log | tail -20
```

## Step 7: Download Results

Once training completes, download checkpoints:

```bash
# On NGC instance
tar -czf v75_checkpoints.tar.gz arc_step_*.pt arc_neural_model_v75.pt

# On your local machine
scp ngc-instance:~/octotetrahedral-agi/v75_checkpoints.tar.gz .
```

## Expected Performance

| Metric | Value |
|--------|-------|
| Training time | 6-24 hours |
| Steps/second | 1-3 (depends on GPU) |
| Memory usage | ~19-20 GB VRAM |
| GPU utilization | 90-100% |

## Troubleshooting

### Out of Memory
If you still hit OOM on NGC:
```bash
# Reduce batch size to 4
--batch-size 4
```

### Slow Training
Check GPU utilization:
```bash
nvidia-smi
# Should show 90-100% GPU usage
# If low, check for I/O bottleneck
```

### Connection Lost
Training runs in background (nohup), so it continues even if SSH disconnects.
Reconnect and check:
```bash
ps aux | grep train_arc
tail -f train_v75_ngc.log
```

## Quick Start Script

Included: `ngc_launch.sh` - automated setup and launch

## Post-Training

1. Download checkpoints to local machine
2. Evaluate with: `python3 arc_solver.py --model arc_neural_model_v75.pt`
3. Generate HTML report: `python3 generate_contest_html_enriched.py`

## Cost Optimization

- Use A100 for best price/performance
- Training should complete in 6-12 hours
- Stop instance immediately after downloading results

## Support

If issues occur:
- Check logs: `train_v75_ngc.log`
- Verify CUDA: `nvidia-smi`
- Check disk space: `df -h`
- Monitor memory: `free -h`
