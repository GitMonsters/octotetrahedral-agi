#!/bin/bash
# ================================================================
# OctoTetrahedral AGI -- Remote GPU Setup
#
# Run this ON the GPU machine (via SSH or as startup script).
# Works on any NVIDIA GPU box: RunPod, Lambda, Colab, bare metal.
#
# Usage (from your laptop):
#   ssh root@<GPU_IP> 'bash -s' < deploy/remote_setup.sh
#
# Or copy and run:
#   scp deploy/remote_setup.sh root@<GPU_IP>:~/setup.sh
#   ssh root@<GPU_IP> 'TRAIN=1 SERVE=1 bash ~/setup.sh'
#
# Environment variables:
#   TRAIN=1       Run ARC training after setup (default: 0)
#   SERVE=1       Start API server after training (default: 0)
#   SCALE=tiny    Model scale preset (tiny|base|large|ultra)
#   EPOCHS=3      Training epochs
#   NUM_TASKS=400 Number of ARC tasks to train on
#   PORT=8000     API server port
# ================================================================

set -euo pipefail

TRAIN="${TRAIN:-0}"
SERVE="${SERVE:-0}"
SCALE="${SCALE:-tiny}"
EPOCHS="${EPOCHS:-3}"
NUM_TASKS="${NUM_TASKS:-400}"
PORT="${PORT:-8000}"
REPO_URL="https://github.com/GitMonsters/octotetrahedral-agi.git"
WORK_DIR="/workspace/octo"

echo "============================================"
echo " OctoTetrahedral AGI -- Remote GPU Setup"
echo " Scale: ${SCALE} | Train: ${TRAIN} | Serve: ${SERVE}"
echo "============================================"
echo ""

# --- 0. GPU Check ---
echo "[0/5] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "GPUs found: ${GPU_COUNT}"
else
    echo "WARNING: nvidia-smi not found. CPU-only mode."
    GPU_COUNT=0
fi
echo ""

# --- 1. Clone repo ---
echo "[1/5] Cloning repository..."
if [ -d "${WORK_DIR}/.git" ]; then
    echo "Repo exists, pulling latest..."
    cd "${WORK_DIR}" && git pull --ff-only
else
    rm -rf "${WORK_DIR}"
    git clone --depth 1 "${REPO_URL}" "${WORK_DIR}"
    cd "${WORK_DIR}"
fi
echo "OK -- repo ready at ${WORK_DIR}"
echo ""

# --- 2. Install dependencies ---
echo "[2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null \
    || pip install --quiet torch  # fallback to default CUDA
pip install --quiet tiktoken pyyaml tqdm numpy Pillow fastapi uvicorn
echo "OK -- dependencies installed"
echo ""

# --- 3. Verify model loads ---
echo "[3/5] Verifying model..."
python3 -c "
import torch
from config import Config
from model import OctoTetrahedralModel
cfg = Config()
cfg.cognitive_geometry.entropy_monitor_enabled = False
cfg.cognitive_geometry.svd_enabled = False
model = OctoTetrahedralModel(cfg, use_geometric_physics=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
params = sum(p.numel() for p in model.parameters())
vram = torch.cuda.memory_allocated()/1e9 if device == 'cuda' else 0
print(f'OK -- {params/1e6:.1f}M params on {device} ({vram:.1f}GB VRAM)')
# Quick forward pass
x = torch.randint(0, 100, (1, 32)).to(device)
with torch.no_grad():
    out = model(input_ids=x)
print(f'OK -- forward pass: logits {out[\"logits\"].shape}')
"
echo ""

# --- 4. Download ARC data + Train ---
if [ "${TRAIN}" = "1" ]; then
    echo "[4/5] Training on ARC-AGI..."
    
    # Get ARC data
    if [ ! -d "arc_data" ]; then
        git clone --depth 1 https://github.com/fchollet/ARC-AGI.git arc_data 2>/dev/null
    fi
    
    python3 -c "
import torch
import torch.nn.functional as F
import numpy as np
import json, random, time, glob

from config import Config
from model import OctoTetrahedralModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = Config()
cfg.cognitive_geometry.entropy_monitor_enabled = False
cfg.cognitive_geometry.svd_enabled = False
model = OctoTetrahedralModel(cfg, use_geometric_physics=False).to(device)

NUM_EPOCHS = ${EPOCHS}
NUM_TASKS = ${NUM_TASKS}
MAX_GRID = 12
LR = 1e-4

train_files = sorted(glob.glob('arc_data/data/training/*.json'))[:NUM_TASKS]
print(f'Training: {len(train_files)} tasks, {NUM_EPOCHS} epochs, grid {MAX_GRID}x{MAX_GRID}')

def grid_to_tensor(grid, max_size=MAX_GRID):
    h, w = len(grid), len(grid[0])
    flat = [v for row in grid for v in row][:max_size*max_size]
    flat = flat + [0] * max(0, max_size*max_size - len(flat))
    tokens = torch.tensor(flat, dtype=torch.long)
    colors = torch.tensor([[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,1,0],
        [.5,.5,.5],[1,0,1],[1,.5,0],[0,1,1],[.5,0,0]])
    img = torch.zeros(3, max_size, max_size)
    for r in range(min(h, max_size)):
        for c in range(min(w, max_size)):
            img[:, r, c] = colors[grid[r][c]]
    return tokens, img

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
history = []
best_loss = float('inf')
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    ep_loss = []
    random.shuffle(train_files)
    for step, f in enumerate(train_files):
        task = json.load(open(f))
        pair = random.choice(task['train'])
        it, ii = grid_to_tensor(pair['input'])
        ot, _ = grid_to_tensor(pair['output'])
        ids = it.unsqueeze(0).to(device)
        imgs = ii.unsqueeze(0).to(device, dtype=torch.float32)
        tgt = ot.unsqueeze(0).to(device)
        
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, images=imgs)
        logits = out['logits']
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        l = loss.item()
        ep_loss.append(l)
        history.append(l)
        if step % 50 == 0:
            elapsed = time.time() - t0
            vram = torch.cuda.memory_allocated()/1e9 if device == 'cuda' else 0
            print(f'  E{epoch+1}/{NUM_EPOCHS} | Step {step:3d}/{len(train_files)} | Loss: {l:.4f} | VRAM: {vram:.1f}GB | {elapsed:.0f}s')
        del ids, imgs, tgt, out, logits, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    avg = np.mean(ep_loss)
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), 'arc_multimodal_best.pt')
        print(f'  -- Saved best checkpoint (loss={best_loss:.4f})')
    print(f'  Epoch {epoch+1} done | Avg: {avg:.4f} | Best: {best_loss:.4f}')

total_time = time.time() - t0
print(f'')
print(f'Training complete in {total_time:.0f}s')
print(f'Loss: {history[0]:.4f} -> {history[-1]:.4f} ({(1-history[-1]/history[0])*100:.1f}% reduction)')
print(f'Checkpoint: arc_multimodal_best.pt')
"
    echo "OK -- training complete"
else
    echo "[4/5] Skipping training (set TRAIN=1 to enable)"
fi
echo ""

# --- 5. Start API server ---
if [ "${SERVE}" = "1" ]; then
    echo "[5/5] Starting API server on port ${PORT}..."
    
    # Kill any existing server
    pkill -f "serve.py" 2>/dev/null || true
    sleep 2
    
    nohup python3 serve.py --scale "${SCALE}" --device cuda:0 --port "${PORT}" \
        > /var/log/octotetrahedral.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
    
    # Wait for it to be ready
    echo "Waiting for server..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "OK -- server ready!"
            curl -sf "http://localhost:${PORT}/health" | python3 -m json.tool
            echo ""
            curl -sf "http://localhost:${PORT}/info" | python3 -m json.tool
            break
        fi
        sleep 2
    done
    
    echo ""
    echo "Endpoints:"
    echo "  Health:   http://$(hostname -I | awk '{print $1}'):${PORT}/health"
    echo "  Info:     http://$(hostname -I | awk '{print $1}'):${PORT}/info"
    echo "  Generate: http://$(hostname -I | awk '{print $1}'):${PORT}/generate"
    echo "  Docs:     http://$(hostname -I | awk '{print $1}'):${PORT}/docs"
    echo ""
    echo "Logs: tail -f /var/log/octotetrahedral.log"
else
    echo "[5/5] Skipping API server (set SERVE=1 to enable)"
fi

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
