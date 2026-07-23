# API Documentation

## Endpoints

### `GET /health`

Returns runtime readiness and accelerator metadata.

Example response:

```json
{
  "status": "healthy",
  "model": "OctoTetrahedralModel",
  "device": "mps",
  "device_backend": "mps",
  "device_fallback_reason": null,
  "mps_available": true,
  "cuda_available": false
}
```

### `POST /predict`

Run inference for a token batch.

Request body:

```json
{
  "input_ids": [12, 42, 77]
}
```

Response body:

```json
{
  "predictions": [[2, 2, 2]],
  "device": "mps",
  "success": true
}
```

## macOS / Metal (MPS) acceleration

The API now treats Apple Metal as a first-class backend:

- prefers CUDA first, then Metal/MPS, then CPU
- validates Metal availability with a smoke test before selecting it
- reports fallback reasons via `GET /health`
- builds request tensors with MPS-safe contiguous `torch.long` inputs
- falls back to CPU automatically when Metal is unavailable during startup

### Platform requirements

- Apple Silicon Mac recommended
- macOS 12.3+
- PyTorch build with MPS support

### Quick validation

```bash
python - <<'PY'
from gpu_support import resolve_device
print(resolve_device("mps").as_dict())
PY
```

### Performance expectations

Use `python -m gpu_support --runs 25 --tokens 128` to generate a comparison table.

Expected Apple Silicon inference profile for medium token batches:

| Device | Avg latency | Throughput | Expected result |
| --- | ---: | ---: | --- |
| CPU | 40-80 ms | 1x baseline | Reference |
| Metal / MPS | 8-16 ms | 5-10x baseline | Target acceleration |

If Metal does not reach the expected range, follow the troubleshooting steps in `METAL_SETUP_GUIDE.md`.
