# OctoTetrahedral AGI — API Documentation

## Overview

The OctoTetrahedral AGI API is a FastAPI-based REST service for neural inference.
It exposes two endpoints: a health check and a prediction endpoint.

- **Base URL**: `http://localhost:8002` (default port)
- **Protocol**: HTTP/1.1
- **Content-Type**: `application/json`
- **Timeout**: 30 s recommended

---

## Endpoints

### `GET /health`

Returns the current health status of the service and device information.

**Request**

```http
GET /health HTTP/1.1
```

**Response** `200 OK`

```json
{
  "status": "healthy",
  "model": "OctoTetrahedralModel",
  "device": "cpu",
  "device_type": "cpu",
  "accelerator": "cpu"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"healthy"` when the server is running |
| `model` | string | Model class name |
| `device` | string | PyTorch device string (`"cpu"`, `"cuda"`, `"mps"`) |
| `device_type` | string | Same as `device` |
| `accelerator` | string | `"cpu"`, `"cuda"`, or `"mps"` |

**cURL Example**

```bash
curl http://localhost:8002/health
```

---

### `POST /predict`

Run inference on a sequence of input token IDs.

**Request Body**

```json
{
  "input_ids": [1, 2, 3, 42, 100]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_ids` | `integer[]` | Yes | Token IDs to process. Each value must be `0 ≤ id ≤ 50000`. Length must be `1 ≤ len ≤ 256`. |

**Response** `200 OK`

```json
{
  "predictions": [0, 1, 0, 2, 1],
  "device": "cpu",
  "success": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | `integer[]` | Predicted token class indices (same length as input) |
| `device` | string | Device used for this inference |
| `success` | bool | Always `true` for 200 responses |

---

## Error Reference

### `400 Bad Request` — Empty Input

```json
{
  "detail": "input_ids must contain at least 1 token."
}
```

Trigger: `input_ids` is an empty list `[]`.

### `400 Bad Request` — Invalid Token ID

```json
{
  "detail": "input_ids[0] must be between 0 and 50000."
}
```

Trigger: Any token ID is negative or greater than 50 000.

### `400 Bad Request` — Wrong Type

```json
{
  "detail": "input_ids[1] must be an integer, got str."
}
```

Trigger: A token ID is a string, float, boolean, or other non-integer type.

### `413 Payload Too Large`

```json
{
  "detail": "input_ids must contain no more than 256 tokens."
}
```

Trigger: `input_ids` contains more than 256 elements.

### `500 Internal Server Error`

```json
{
  "detail": "..."
}
```

Trigger: Unexpected inference failure (model error, out-of-memory, etc.).

---

## Request Limits

| Limit | Value |
|-------|-------|
| Min tokens | 1 |
| Max tokens | 256 |
| Min token ID | 0 |
| Max token ID | 50 000 |
| Max request body | ~10 KB (256 × 4 bytes + JSON overhead) |
| Timeout (recommended) | 30 s |

---

## Examples

### cURL

```bash
# Health check
curl http://localhost:8002/health

# Single token
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d '{"input_ids": [42]}'

# Normal batch
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d '{"input_ids": [1, 2, 3, 4, 5]}'

# Empty input → 400
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d '{"input_ids": []}'

# Large batch → 413
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d "{\"input_ids\": [$(seq -s, 1 1000)]}"
```

### Python (httpx)

```python
import httpx

client = httpx.Client(base_url="http://localhost:8002", timeout=30.0)

# Health check
health = client.get("/health").json()
print(health["status"])  # "healthy"

# Prediction
response = client.post("/predict", json={"input_ids": [1, 2, 3, 42]})
if response.status_code == 200:
    print(response.json()["predictions"])
elif response.status_code == 400:
    print("Validation error:", response.json()["detail"])
elif response.status_code == 413:
    print("Too many tokens:", response.json()["detail"])
```

### Python (requests)

```python
import requests

r = requests.post(
    "http://localhost:8002/predict",
    json={"input_ids": [10, 20, 30]},
    timeout=30,
)
r.raise_for_status()
print(r.json()["predictions"])
```

### JavaScript (fetch)

```javascript
const response = await fetch("http://localhost:8002/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input_ids: [1, 2, 3] }),
  signal: AbortSignal.timeout(30_000),
});

if (!response.ok) {
  const err = await response.json();
  throw new Error(`${response.status}: ${err.detail}`);
}

const { predictions } = await response.json();
console.log(predictions);
```

---

## Performance Specifications

| Metric | CPU | Metal (MPS) | CUDA |
|--------|-----|-------------|------|
| p50 latency | ~65 ms | ~7 ms | ~7 ms |
| p99 latency | ~70 ms | ~13 ms | ~13 ms |
| Throughput | 16 req/s | 150+ req/s | 155+ req/s |

See [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md) for full details.

---

## macOS / Metal Support

When running on Apple Silicon (M1/M2/M3), the API automatically uses the
Metal/MPS backend for ~8× speedup:

```bash
# Enable Metal explicitly
export OCTO_DEVICE=mps
python api.py

# Verify Metal is active
curl http://localhost:8002/health
# → "accelerator": "mps"
```

If Metal inference fails at runtime, the API automatically retries on CPU.

See [METAL_SETUP_GUIDE.md](METAL_SETUP_GUIDE.md) for setup instructions.

---

## Best Practices

### Batch Size
- Use the full 256-token window for maximum throughput
- For real-time responses, keep batches small (≤ 64 tokens) to stay within latency targets

### Retry Logic

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def predict(input_ids: list[int]) -> list[int]:
    r = httpx.post(
        "http://localhost:8002/predict",
        json={"input_ids": input_ids},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["predictions"]
```

### Caching
For repeated identical inputs, cache at the application level to reduce latency to near zero.

### Monitoring
Poll `GET /health` every 30 s. Alert if `status != "healthy"` or if the endpoint is unreachable.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `500` on startup | Checkpoint not found | Place checkpoint at `checkpoints/arc/arc_final.pt` |
| `500` on inference | Out-of-memory | Reduce concurrency; use GPU |
| Very high latency | CPU under load | Enable GPU acceleration |
| `400` on valid input | Token ID out of range | Check values are `0–50000` |

---

*OctoTetrahedral AGI v1.0.0 | [GitHub](https://github.com/GitMonsters/octotetrahedral-agi)*
