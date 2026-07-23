# API Documentation

## Overview

The OctoTetrahedral AGI inference API exposes two endpoints:

- `GET /health`
- `POST /predict`

## Performance Specifications

- Average latency: **65.29 ms**
- Throughput: **16.20 req/s** at **50 concurrent requests**
- Request size limits: **1-256 tokens**
- Timeout guidance: **30 seconds**
- Content type: **application/json**

## `GET /health`

Returns service readiness and active device information.

### Example Response

```json
{
  "status": "healthy",
  "model": "OctoTetrahedralModel",
  "device": "cpu",
  "requested_device": "auto",
  "accelerator": null,
  "fallback_used": true,
  "expected_speedup_factor": 10.0
}
```

## `POST /predict`

Runs inference for a single token batch.

### Request Schema

```json
{
  "input_ids": [100, 200, 300]
}
```

### Request Rules

- `input_ids` must be a JSON array
- Minimum length: **1**
- Maximum length: **256**
- Token IDs must be integers between **0** and **50000**

### Example Success Response

```json
{
  "predictions": [[444, 444, 444]],
  "device": "cpu",
  "success": true
}
```

## Error Handling

| Status | When it happens | Example detail |
| --- | --- | --- |
| `400 Bad Request` | Empty input, non-integer token, out-of-range token | `input_ids must contain at least 1 token.` |
| `413 Payload Too Large` | More than 256 tokens | `input_ids must contain no more than 256 tokens.` |
| `500 Internal Server Error` | Model loading or inference failure | Underlying exception message |

## Best Practices

- Keep batches between **5 and 50 tokens** for predictable CPU latency.
- Retry only idempotent requests and use exponential backoff for transient `500` responses.
- Cache repeated inference inputs when request duplication is common.
- Monitor p95/p99 latency, error rate, and memory usage during sustained load.
- Enable GPU acceleration for production traffic spikes or concurrency above the 50-request CPU baseline.

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_ids":[100,200,300]}'
```

### Python (`requests`)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"input_ids": [100, 200, 300]},
    timeout=30,
)
print(response.json())
```

### JavaScript (`fetch`)

```javascript
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ input_ids: [100, 200, 300] }),
});

console.log(await response.json());
```

## GPU Setup

- **CUDA**: install a CUDA-enabled PyTorch build and set `OCTO_DEVICE=cuda`.
- **Metal / Apple Silicon**: use a PyTorch build with MPS support and set `OCTO_DEVICE=mps` or leave auto-detect enabled.
- If the requested accelerator is unavailable or fails a smoke test, the API falls back to **CPU** automatically.

## Troubleshooting

- **`400` responses**: verify token count and integer range.
- **`413` responses**: split large requests into smaller batches.
- **Slow throughput**: enable GPU support, add caching, or reduce batch size.
- **Unexpected CPU usage**: check `GET /health` to confirm whether accelerator fallback occurred.
