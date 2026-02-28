# OctoTetrahedral AGI — Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 IONOS Cloud (EU)                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  H100/H200 GPU VM                            │   │
│  │  ┌────────────────────────────────────────┐  │   │
│  │  │  Docker + NVIDIA Container Runtime      │  │   │
│  │  │  ┌──────────────────────────────────┐  │  │   │
│  │  │  │  FastAPI (serve.py)              │  │  │   │
│  │  │  │  ├─ /health                      │  │  │   │
│  │  │  │  ├─ /info                        │  │  │   │
│  │  │  │  ├─ /generate                    │  │  │   │
│  │  │  │  └─ /generate/stream (SSE)       │  │  │   │
│  │  │  │                                  │  │  │   │
│  │  │  │  OctoTetrahedral MoE Model       │  │  │   │
│  │  │  │  (7B / 70B / 1.72T)              │  │  │   │
│  │  │  └──────────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Provision IONOS GPU VM

- Log into [IONOS Cloud Console](https://dcd.ionos.com)
- Create a **Cloud GPU VM** → choose **H100 80GB** (or H200 141GB)
- For 7B MoE: **1× GPU** is sufficient
- For 70B MoE: **4-8× GPUs** (XL config)
- OS: Ubuntu 22.04 (NVIDIA drivers pre-installed)

### 2. Deploy

SSH into your VM and run:

```bash
# One-liner setup
curl -sSL https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/deploy/ionos_setup.sh | bash

# Or with a specific model config
MODEL_CONFIG=70b bash deploy/ionos_setup.sh
```

### 3. Test

```bash
# Health check
curl http://<your-ip>:8000/health

# Model info
curl http://<your-ip>:8000/info

# Generate text
curl -X POST http://<your-ip>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The OctoTetrahedral architecture", "max_tokens": 100}'

# Interactive docs
open http://<your-ip>:8000/docs
```

## Model Configs

| Config | Total Params | Active/Token | IONOS VM | Est. Cost |
|--------|-------------|-------------|----------|-----------|
| `7b`   | 6.8B        | 2.6B        | 1× H100 (S) | ~$3/hr |
| `70b`  | 70.6B       | 22.3B       | 4-8× H100 (XL) | ~$20-40/hr |
| `1.72t`| 1.72T       | 275B        | Multi-node (not single-server) | Contact IONOS |

## Using NVIDIA Dev Account

### Free NIM Prototyping (no GPU needed)

Your NVIDIA dev account gives free access to NIM API endpoints:

```bash
# Test via NVIDIA's DGX Cloud-hosted endpoints
curl https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Self-Hosted NIM Container on IONOS

```bash
# Pull NIM container (free for dev, up to 16 GPUs)
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key from developer.nvidia.com>

# Run NIM alongside OctoTetrahedral for hybrid inference
docker run --gpus all -p 8001:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### TensorRT-LLM Optimization (2-4× faster inference)

```bash
# Convert model to TensorRT-LLM format for production speed
pip install tensorrt-llm
# Export and optimize (future work — requires custom converter)
```

## Files

```
serve.py                  — FastAPI inference server
deploy/
  Dockerfile.gpu          — NVIDIA GPU container image
  ionos_setup.sh          — One-click IONOS VM provisioner
  DEPLOY.md               — This file
configs/
  octo_7b_moe.py          — 7B MoE config (single GPU)
  octo_70b_moe.py         — 70B MoE config (multi-GPU)
  octo_1_72t.py           — 1.72T MoE config (multi-node)
train_distributed.py      — FSDP distributed training
```

## Security Notes

- Add a reverse proxy (nginx/caddy) with TLS for production
- Restrict port 8000 to your IP in IONOS firewall rules
- Set `NVIDIA_API_KEY` as environment variable, not in code
