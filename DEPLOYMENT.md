# OctoTetrahedral AGI — Deployment Guide

## Quick Start (5-minute setup)

```bash
# 1. Clone the repository
git clone https://github.com/GitMonsters/octotetrahedral-agi.git
cd octotetrahedral-agi

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
pip install -r requirements.txt
pip install fastapi uvicorn

# 3. Start the API
python api.py
# → Server running at http://localhost:8002

# 4. Test the API
curl http://localhost:8002/health
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d '{"input_ids": [1, 2, 3]}'
```

---

## System Requirements

### macOS — Metal (Apple Silicon)

| Component | Requirement |
|-----------|------------|
| **Chip** | Apple M1 / M2 / M3 (any variant) |
| **macOS** | 12.3+ (Monterey or later) |
| **Python** | 3.9+ |
| **PyTorch** | 2.0+ (Metal/MPS backend included) |
| **Memory** | 8 GB unified memory (16 GB recommended) |
| **Disk** | 3 GB free |

```bash
# macOS Metal install
pip install torch torchvision
pip install -r requirements.txt fastapi uvicorn

# Verify Metal backend
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True

# Enable Metal
export OCTO_DEVICE=mps
python api.py
```

### Linux — CUDA (NVIDIA GPU)

| Component | Requirement |
|-----------|------------|
| **GPU** | NVIDIA with compute capability ≥ 6.0 (Pascal+) |
| **CUDA** | 11.8+ |
| **cuDNN** | 8.6+ |
| **Python** | 3.9+ |
| **VRAM** | 8 GB (16 GB recommended) |
| **RAM** | 16 GB |

```bash
# CUDA install (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt fastapi uvicorn

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Enable CUDA
export OCTO_DEVICE=cuda
python api.py
```

### Windows — CUDA Support

```powershell
# Windows (PowerShell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt fastapi uvicorn

$env:OCTO_DEVICE = "cuda"
python api.py
```

### CPU-Only (Any Platform)

```bash
# Minimum: Python 3.9+, 8 GB RAM
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt fastapi uvicorn

# No environment variable needed — CPU is the default fallback
python api.py
```

---

## Installation Methods

### Docker / Container Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install PyTorch (CPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy application
COPY . .

EXPOSE 8002
CMD ["python", "api.py", "8002"]
```

```bash
# Build and run
docker build -t octotetrahedral-agi:1.0.0 .
docker run -p 8002:8002 octotetrahedral-agi:1.0.0

# With GPU (NVIDIA)
docker run --gpus all -p 8002:8002 \
  -e OCTO_DEVICE=cuda \
  octotetrahedral-agi:1.0.0
```

#### Docker Compose

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8002:8002"
    environment:
      - OCTO_DEVICE=cpu          # Change to cuda or mps as needed
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Local Development Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/GitMonsters/octotetrahedral-agi.git
cd octotetrahedral-agi

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Install dev dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install fastapi uvicorn httpx

# 4. Run tests
python -m pytest tests/test_api.py tests/test_gpu_support.py -v

# 5. Start the server (with auto-reload)
uvicorn api:app --reload --port 8002
```

### Cloud Deployment — AWS

#### AWS EC2 (g4dn.xlarge — NVIDIA T4)

```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-0c94855ba95b798ca \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups your-sg

# SSH in and install
ssh ec2-user@<ip>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/GitMonsters/octotetrahedral-agi.git
cd octotetrahedral-agi && pip install -r requirements.txt fastapi uvicorn
export OCTO_DEVICE=cuda
python api.py 8002
```

#### AWS ECS / Fargate

```json
{
  "family": "octotetrahedral-agi",
  "containerDefinitions": [{
    "name": "api",
    "image": "your-ecr-repo/octotetrahedral-agi:1.0.0",
    "portMappings": [{"containerPort": 8002}],
    "memory": 4096,
    "cpu": 2048,
    "environment": [{"name": "OCTO_DEVICE", "value": "cpu"}],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3
    }
  }]
}
```

### Cloud Deployment — GCP

```bash
# Cloud Run (serverless)
gcloud run deploy octotetrahedral-agi \
  --image gcr.io/your-project/octotetrahedral-agi:1.0.0 \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --port 8002 \
  --set-env-vars OCTO_DEVICE=cpu \
  --min-instances 1

# GKE (Kubernetes Engine)
kubectl apply -f deploy/k8s/
```

### Cloud Deployment — Azure

```bash
# Azure Container Instances
az container create \
  --resource-group myRG \
  --name octotetrahedral-agi \
  --image your-acr.azurecr.io/octotetrahedral-agi:1.0.0 \
  --cpu 2 --memory 4 \
  --ports 8002 \
  --environment-variables OCTO_DEVICE=cpu
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: octotetrahedral-agi
spec:
  replicas: 3
  selector:
    matchLabels:
      app: octotetrahedral-agi
  template:
    metadata:
      labels:
        app: octotetrahedral-agi
    spec:
      containers:
      - name: api
        image: octotetrahedral-agi:1.0.0
        ports:
        - containerPort: 8002
        env:
        - name: OCTO_DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: octotetrahedral-agi-service
spec:
  selector:
    app: octotetrahedral-agi
  ports:
  - port: 80
    targetPort: 8002
  type: LoadBalancer
```

---

## Configuration

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `OCTO_DEVICE` | `cpu`, `cuda`, `mps` | auto-detect | Override device selection |
| `OCTOTETRAHEDRAL_DEVICE` | same as above | auto-detect | Alternative device override |
| `CUDA_VISIBLE_DEVICES` | `""` to disable | system default | Disable CUDA entirely |

### Device Selection

The API automatically selects the best available device at startup:

```
1. Check OCTO_DEVICE / OCTOTETRAHEDRAL_DEVICE env var
2. Try CUDA (with smoke test)
3. Try Metal/MPS (with smoke test)
4. Fall back to CPU
```

Force a specific device:

```bash
# CPU
export OCTO_DEVICE=cpu

# CUDA (NVIDIA)
export OCTO_DEVICE=cuda

# Metal (Apple Silicon)
export OCTO_DEVICE=mps

# Disable CUDA (use MPS or CPU)
export CUDA_VISIBLE_DEVICES=""
```

### Performance Tuning

```python
# torch.compile (PyTorch 2.0+) — optional, reduces latency by 10–20%
import torch
model = torch.compile(model)

# Thread control for CPU deployments
torch.set_num_threads(4)         # Limit CPU threads
torch.set_num_interop_threads(2) # Limit inter-op threads
```

### Resource Limits

| Limit | Value | Configurable |
|-------|-------|-------------|
| Max input tokens | 256 | Edit `MAX_INPUT_TOKENS` in `api.py` |
| Min token ID | 0 | Edit `MIN_TOKEN_ID` in `api.py` |
| Max token ID | 50 000 | Edit `MAX_TOKEN_ID` in `api.py` |
| Request timeout | 30 s | Configure in reverse proxy/load balancer |

---

## Monitoring & Debugging

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8002/health

# Response
{
  "status": "healthy",
  "model": "OctoTetrahedralModel",
  "device": "cpu",
  "device_type": "cpu",
  "accelerator": "cpu"
}
```

### Performance Metrics

The API logs structured output at `INFO` level:

```
INFO:     ✅ Model loaded on cpu
INFO:     ✅ Prediction successful
INFO:     ❌ Inference error: ...
```

For production, pipe these logs to your observability stack (Datadog, CloudWatch, GCP Logging, etc.).

### Prometheus / Grafana Integration

Add `prometheus-fastapi-instrumentator` for metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Then scrape `http://localhost:8002/metrics`.

### Error Handling

| HTTP Code | Meaning | Common Cause |
|-----------|---------|--------------|
| 200 | Success | Normal inference |
| 400 | Bad Request | Empty/invalid input |
| 413 | Payload Too Large | `input_ids` > 256 tokens |
| 500 | Internal Server Error | Inference failure |
| 503 | Service Unavailable | Server starting up |

### Troubleshooting Guide

#### Model checkpoint not found

```
Error: Failed to load model: [Errno 2] No such file or directory: 'checkpoints/arc/arc_final.pt'
```

**Fix**: Place your checkpoint at `checkpoints/arc/arc_final.pt`, or update the path in `api.py`.

#### CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Fix**:
1. Reduce batch size
2. Add `torch.cuda.empty_cache()` calls
3. Use a GPU with more VRAM
4. Fall back to CPU: `export OCTO_DEVICE=cpu`

#### MPS / Metal errors on Apple Silicon

```
RuntimeError: MPS backend out of memory
```

**Fix**:
1. Set `export OCTO_DEVICE=cpu` to fall back
2. Close other GPU-using applications
3. Check `python api.py` — it auto-falls back to CPU on Metal failure

#### High latency on first request

This is expected — the model warms up on first use. Subsequent requests are faster.

---

## Production Checklist

### Pre-deployment Validation

- [ ] Checkpoint file exists and loads without error
- [ ] `GET /health` returns `{"status": "healthy"}`
- [ ] Valid input returns 200 with predictions
- [ ] Empty input returns 400
- [ ] 1000-token batch returns 413
- [ ] Out-of-range token ID returns 400

### Load Testing Validation

```bash
# Install wrk
# Test sustained load
wrk -t4 -c50 -d30s -s scripts/wrk_predict.lua http://localhost:8002/predict

# Acceptance criteria:
# - Error rate < 1%
# - p99 latency < 200 ms (CPU) or < 25 ms (GPU)
# - No memory leaks (RSS stable after 1 min)
```

### Performance Targets

| Metric | CPU Target | GPU Target |
|--------|-----------|-----------|
| p50 latency | < 70 ms | < 10 ms |
| p99 latency | < 100 ms | < 20 ms |
| Throughput | ≥ 15 req/s | ≥ 100 req/s |
| Error rate | < 0.1% | < 0.1% |
| Memory (RSS) | < 4 GB | < 3 GB |

### Monitoring Setup

- [ ] Log aggregation configured (stdout → log service)
- [ ] `GET /health` in load-balancer health check
- [ ] Alerts on error rate > 1%
- [ ] Alerts on p99 latency > 2× target
- [ ] Memory usage monitored (cgroup or OS-level)
- [ ] Disk usage monitored (checkpoint + logs)

---

## Resources

- [OctoTetrahedral AGI README](README.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Performance Comparison](PERFORMANCE_COMPARISON.md)
- [Metal Setup Guide](METAL_SETUP_GUIDE.md)
- [Benchmark Report](BENCHMARK_REPORT.md)
- [PyTorch MPS documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch CUDA documentation](https://pytorch.org/docs/stable/cuda.html)
