# OctoTetrahedral AGI — Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│            GPU Cloud (Lambda / RunPod / IONOS)       │
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

## Quick Start — Choose Your Cloud

### Option A: Lambda Labs (Recommended — $2.49/hr H100, no egress fees)

```bash
# 1. Sign up at https://lambda.ai
# 2. Get API key from https://cloud.lambdalabs.com/api-keys
export LAMBDA_API_KEY="your-key"
bash deploy/provision_lambda.sh
```

### Option B: RunPod ($1.99-2.39/hr H100)

```bash
# 1. Sign up at https://runpod.io
# 2. Get API key from Settings → API Keys
export RUNPOD_API_KEY="your-key"
bash deploy/provision_runpod.sh
```

### Option C: IONOS Cloud (EU, H200 available)

> ⚠️ Requires IONOS Cloud Unlimited Access approval.

```bash
# 1. Get token from DCD → Management → Token Manager
export IONOS_TOKEN="your-token"
bash deploy/provision_ionos.sh
```

### 1. Provision GPU VM

- **Lambda Labs**: 1× H100 SXM 80GB, Ubuntu pre-installed, SSH ready
- **RunPod**: 1× H100 80GB pod, PyTorch pre-installed
- **IONOS**: H100/H200 Cloud GPU VM, Ubuntu 22.04
- For 7B MoE: **1× GPU** is sufficient
- For 70B MoE: **4-8× GPUs**
- **Firewall**: Allow port 443 (HTTPS) + 22 (SSH from your IP). Block 8000.

### 2. DNS Setup

Point `api.transcendplexity.ai` to your VM's public IP:
- IONOS Domains → `transcendplexity.ai` → DNS → Add A Record:
  - Host: `api`
  - Points to: `<your VM IP>`
  - TTL: 3600

### 3. Deploy

SSH into your VM and run:

```bash
# Full deploy with TLS + API auth
DOMAIN=api.transcendplexity.ai \
OCTO_API_KEYS=your-secret-key-here \
MODEL_CONFIG=7b \
bash <(curl -sSL https://raw.githubusercontent.com/GitMonsters/octotetrahedral-agi/main/deploy/ionos_setup.sh)
```

### 4. Test

```bash
# Health check
curl https://api.transcendplexity.ai/health

# Model info
curl https://api.transcendplexity.ai/info

# Generate text
curl -X POST https://api.transcendplexity.ai/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{"prompt": "The OctoTetrahedral architecture", "max_tokens": 100}'

# Streaming
curl -N -X POST https://api.transcendplexity.ai/generate/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key-here" \
  -d '{"prompt": "Hello", "max_tokens": 50}'

# Metrics (internal only)
curl http://localhost:8000/metrics

# Interactive docs
open https://api.transcendplexity.ai/docs
```

## Model Configs

| Config | Total Params | Active/Token | GPU Requirement | Lambda Labs | RunPod |
|--------|-------------|-------------|-----------------|-------------|--------|
| `7b`   | 6.8B        | 2.6B        | 1× H100 80GB   | ~$2.49/hr   | ~$1.99/hr |
| `70b`  | 70.6B       | 22.3B       | 4-8× H100      | ~$10-20/hr  | ~$8-16/hr |
| `1.72t`| 1.72T       | 275B        | Multi-node cluster | Contact provider | N/A |

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
serve.py                  — FastAPI inference server (auth + metrics)
train_arc_moe.py          — ARC training pipeline (real + synthetic data)
eval_arc_moe.py           — ARC-AGI benchmark evaluation
train_distributed.py      — FSDP distributed training
finetune_targeted.py      — Targeted fine-tuning on weak ARC categories
nim_export.py             — NVIDIA NIM export + TensorRT optimization
deploy/
  Dockerfile.gpu          — NVIDIA GPU container image
  provision_lambda.sh     — Lambda Labs H100 provisioner (recommended)
  provision_runpod.sh     — RunPod H100 provisioner
  provision_ionos.sh      — IONOS Cloud GPU provisioner
  ionos_setup.sh          — One-click VM setup (nginx + TLS, cloud-agnostic)
  nginx.conf              — Reverse proxy with rate limiting
  DEPLOY.md               — This file
configs/
  octo_7b_moe.py          — 7B MoE config (single GPU)
  octo_70b_moe.py         — 70B MoE config (multi-GPU)
  octo_1_72t.py           — 1.72T MoE config (multi-node)
```

## Domain

Primary API domain: `api.transcendplexity.ai`

Available domains:
- transcendplexity.ai (expires 11/2027) — primary
- transcendplexity.com/.org/.io/.tech/.info/.store (expire 11/2026)

## Security Notes

- TLS is auto-configured via Let's Encrypt when `DOMAIN` is set
- API keys required when `OCTO_API_KEYS` env var is set (comma-separated for multiple keys)
- nginx rate limits at 10 req/s per IP (burst 20)
- `/metrics` endpoint restricted to localhost only
- Restrict port 8000 in IONOS firewall — only expose 443 (HTTPS) + 22 (SSH)
- Set `NVIDIA_API_KEY` as environment variable, not in code
