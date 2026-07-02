# Production Deployment Guide: Unified Cognitive Stack Inference Pipeline

## Overview

This guide covers deploying the unified cognitive stack as a production-ready AGI inference service.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                        │
│                                                             │
│  InferenceRequest ──► InferenceService ──► UnifiedForward   │
│                           │                    Model         │
│                           ▼                                 │
│                    InferenceMonitor ◄── coherence / timing  │
│                           │                                 │
│                    InferenceResponse                        │
└─────────────────────────────────────────────────────────────┘
```

**Key modules:**

| Module | Purpose |
|--------|---------|
| `production_config.py` | All tunable parameters and env-specific defaults |
| `api_types.py` | TypedDict request/response types + JSON serialization |
| `inference_service.py` | Core service: pool, retry, timeout, fallback |
| `monitoring.py` | Coherence tracking, alerting, histograms, timing |
| `health_check.py` | Self-test suite, diagnostics report |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCTOAGI_ENV` | `dev` | Environment: `dev`, `staging`, `prod` |
| `OCTOAGI_MODEL_VERSION` | `1.0.0` | Model version tag |
| `OCTOAGI_MODEL_PATH` | _(empty)_ | Path to serialized model (empty = in-memory) |
| `OCTOAGI_LIMB_COUNT` | `8` | Number of cognitive limbs |
| `OCTOAGI_INFERENCE_TIMEOUT_MS` | `20.0` | Per-inference timeout in milliseconds |
| `OCTOAGI_MAX_RETRIES` | `3` | Maximum retry attempts on transient failure |
| `OCTOAGI_POOL_SIZE` | `4` | Model instance pool size |
| `OCTOAGI_COHERENCE_THRESHOLD` | `0.90` | Minimum coherence before alert fires |
| `OCTOAGI_LATENCY_WARN_MS` | `20.0` | Latency threshold for warning log |
| `OCTOAGI_LOG_LEVEL` | env-specific | `DEBUG` / `INFO` / `WARNING` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run health check

```bash
python - <<'EOF'
from health_check import run_health_check
import json
status = run_health_check()
print(json.dumps(status, indent=2))
EOF
```

Expected output:

```json
{
  "healthy": true,
  "model_loaded": true,
  "coherence_baseline": 0.99,
  "limb_symmetry_ok": true,
  "self_test_passed": true,
  ...
}
```

### 3. Run a single inference

```python
from api_types import make_request
from inference_service import InferenceService

service = InferenceService()
req = make_request([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], task_signal="reasoning")
resp = service.infer(req)
print(resp["coherence"], resp["action_channel"])
```

### 4. Run a batch

```python
from api_types import make_batch_request, make_request
from inference_service import InferenceService

service = InferenceService()
requests = [make_request([0.5] * 8, task_signal=f"task-{i}") for i in range(10)]
batch = make_batch_request(requests)
result = service.infer_batch(batch)
print(f"Processed {len(result['responses'])} inferences in {result['total_latency_ms']:.1f} ms")
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OCTOAGI_ENV=prod
ENV OCTOAGI_POOL_SIZE=8
ENV OCTOAGI_COHERENCE_THRESHOLD=0.90

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "from health_check import run_health_check; import sys; s=run_health_check(); sys.exit(0 if s['healthy'] else 1)"

CMD ["python", "serve.py"]
```

### docker-compose

```yaml
version: "3.9"
services:
  inference:
    build: .
    ports:
      - "8080:8080"
    environment:
      OCTOAGI_ENV: prod
      OCTOAGI_POOL_SIZE: "8"
      OCTOAGI_COHERENCE_THRESHOLD: "0.90"
    healthcheck:
      test: ["CMD", "python", "-c", "from health_check import run_health_check; import sys; s=run_health_check(); sys.exit(0 if s['healthy'] else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: octoagi-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: octoagi-inference
  template:
    metadata:
      labels:
        app: octoagi-inference
    spec:
      containers:
        - name: inference
          image: octoagi-inference:latest
          env:
            - name: OCTOAGI_ENV
              value: "prod"
            - name: OCTOAGI_POOL_SIZE
              value: "4"
            - name: OCTOAGI_COHERENCE_THRESHOLD
              value: "0.90"
          livenessProbe:
            exec:
              command:
                - python
                - -c
                - "from health_check import run_health_check; import sys; s=run_health_check(num_tests=1); sys.exit(0 if s['healthy'] else 1)"
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            exec:
              command:
                - python
                - -c
                - "from health_check import run_health_check; import sys; s=run_health_check(num_tests=1); sys.exit(0 if s['healthy'] else 1)"
            initialDelaySeconds: 5
            periodSeconds: 10
```

---

## Health Check Endpoint

Integrate `health_check.run_health_check()` into your HTTP server:

```python
# Example: FastAPI
from fastapi import FastAPI
from health_check import run_health_check

app = FastAPI()

@app.get("/health")
def health():
    return run_health_check()

@app.get("/ready")
def ready():
    status = run_health_check(num_tests=1)
    if not status["healthy"]:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=status)
    return {"status": "ok"}
```

---

## Monitoring Dashboards

### Key Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|----------------|
| Coherence (mean) | `monitor.stats()["mean_coherence"]` | `< 0.90` |
| Coherence (min) | `monitor.stats()["min_coherence"]` | `< 0.85` |
| p99 Latency | `monitor.stats()["p99_latency_ms"]` | `> 20 ms` |
| Alert count | `monitor.stats()["alert_count"]` | `> 0` |
| Limb activation | `monitor.stats()["limb_activation_histogram"]` | uneven distribution |

### Prometheus-compatible scrape (example)

```python
from monitoring import InferenceMonitor

_monitor = InferenceMonitor()  # shared singleton

def metrics_handler():
    s = _monitor.stats()
    lines = [
        f'octoagi_coherence_mean {s["mean_coherence"]}',
        f'octoagi_coherence_min {s["min_coherence"]}',
        f'octoagi_latency_p99_ms {s["p99_latency_ms"]}',
        f'octoagi_alert_count {s["alert_count"]}',
        f'octoagi_total_inferences {s["total_inferences"]}',
    ]
    return "\n".join(lines)
```

---

## Performance Targets

| Target | Value |
|--------|-------|
| p99 inference latency | < 20 ms |
| Coherence floor | ≥ 0.90 |
| Batch size range | 1–100 |
| Uptime target | 99.9% |
| Pool size (prod) | 4–8 instances |

---

## Troubleshooting

### Coherence drops below 0.90

1. Check `monitor.recent_alerts()` for the affected request IDs.
2. Inspect `limb_activation_histogram` for limb imbalance.
3. Verify input `limb_states` are in a reasonable range (0.0–1.0).
4. Reduce batch size to isolate task signals causing degradation.

### High p99 latency

1. Increase `OCTOAGI_POOL_SIZE` to reduce pool wait time.
2. Check `OCTOAGI_INFERENCE_TIMEOUT_MS` is not too low (causing unnecessary retries).
3. Profile with `benchmarks/unified_perf.py` to establish baseline.

### Health check fails

1. Run `run_health_check(num_tests=5)` and inspect `self_test_details`.
2. Check `diagnostics["load_error"]` for model loading issues.
3. Confirm Python path includes the repository root.

### `model pool exhausted` errors

1. Raise `OCTOAGI_POOL_SIZE`.
2. Reduce concurrency at the load-balancer level.
3. Check for stuck model instances via `service._pool.size()`.

---

## Running Tests

```bash
# All unified stack tests
python -m pytest -q tests/test_unified.py

# Production pipeline integration tests (8 tests)
python -m pytest -q tests/test_production_pipeline.py

# Full test suite
python -m pytest -q tests/

# Performance benchmark
python -m benchmarks.unified_perf
```
