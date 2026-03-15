# OctoTetrahedral AGI -- Deployment Guide

## Quick Deploy (One Command)

```bash
# RunPod H100 (~$1.99/hr) -- provisions, trains, serves
export RUNPOD_API_KEY="your-key"
bash deploy/deploy.sh runpod

# Lambda Labs H100 (~$2.49/hr)
export LAMBDA_API_KEY="your-key"
bash deploy/deploy.sh lambda

# Existing SSH box
bash deploy/deploy.sh root@209.20.159.131
```

Options via environment variables:
```bash
SCALE=tiny TRAIN=1 SERVE=1 EPOCHS=3 NUM_TASKS=400 bash deploy/deploy.sh runpod
```

## Architecture

```
GPU Cloud (Lambda / RunPod / any SSH box)
  +-- H100/H200 GPU VM
       +-- OctoTetrahedralModel (160M-1.75T params)
       |   +-- 14-stream CompoundBraid (genus-13)
       |   +-- Vision + Audio + Embodiment encoders
       |   +-- MoE + SOA hybrid routing
       +-- FastAPI (serve.py)
            +-- /health, /info, /generate, /docs
```

## Scale Presets

| Preset | Total Params | Active/Token | GPUs Needed | Cost/hr |
|--------|-------------|-------------|-------------|---------|
| tiny   | 204M        | 204M        | 1x T4/A100  | ~$0.50  |
| base   | 26.2B       | 3.1B        | 4x H100     | ~$8-10  |
| large  | 1.66T       | 99.3B       | 207x H100   | cluster |
| ultra  | 1.75T       | 87.4B       | 219x H100   | cluster |

## Manual Deploy (Step by Step)

### 1. Provision GPU

**RunPod:**
```bash
export RUNPOD_API_KEY="your-key"
bash deploy/provision_runpod.sh
```

**Lambda Labs:**
```bash
export LAMBDA_API_KEY="your-key"
bash deploy/provision_lambda.sh
```

### 2. Setup + Train + Serve

SSH into your box and run:
```bash
# Clone + install + verify
git clone https://github.com/GitMonsters/octotetrahedral-agi.git /workspace/octo
cd /workspace/octo
pip install tiktoken pyyaml tqdm numpy Pillow fastapi uvicorn

# Train on ARC
TRAIN=1 SERVE=0 bash deploy/remote_setup.sh

# Start API
python serve.py --scale tiny --device cuda:0 --port 8000
```

Or one-shot via SCP:
```bash
scp deploy/remote_setup.sh root@<GPU_IP>:~/setup.sh
ssh root@<GPU_IP> 'TRAIN=1 SERVE=1 bash ~/setup.sh'
```

### 3. Docker Deploy

```bash
docker build -f deploy/Dockerfile.gpu -t octotetrahedral:multimodal .
docker run --gpus all -p 8000:8000 octotetrahedral:multimodal
```

### 4. Test API

```bash
curl http://<GPU_IP>:8000/health
curl http://<GPU_IP>:8000/info
curl -X POST http://<GPU_IP>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The OctoTetrahedral architecture", "max_tokens": 100}'
```

### 5. DNS (Optional)

Point `api.transcendplexity.ai` -> your VM IP (A record, TTL 3600).

## Colab (Free GPU)

Open directly:
```
https://colab.research.google.com/github/GitMonsters/octotetrahedral-agi/blob/main/deploy/OctoTetrahedral_MultiModal_Demo.ipynb
```
Runtime > Run all. Tesla T4, 15GB VRAM, ~10 min.

## Files

```
serve.py                  -- FastAPI inference server
deploy/
  deploy.sh               -- One-command deploy (any provider)
  remote_setup.sh          -- Remote GPU setup script  
  Dockerfile.gpu           -- NVIDIA GPU container
  provision_lambda.sh      -- Lambda Labs provisioner
  provision_runpod.sh      -- RunPod provisioner
  provision_ionos.sh       -- IONOS provisioner
  launch.sh                -- IONOS one-shot launcher
  ionos_setup.sh           -- VM setup (nginx + TLS)
  nginx.conf               -- Reverse proxy config
  OctoTetrahedral_*.ipynb  -- Colab notebook
  quickstart.sh            -- Quick reference
```

## Domain

Primary: `api.transcendplexity.ai`
Available: transcendplexity.ai/.com/.org/.io/.tech/.info/.store
