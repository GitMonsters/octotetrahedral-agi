# Metal Setup Guide

## Requirements

- Apple Silicon Mac
- macOS 12.3 or newer
- Python 3.11+
- PyTorch 2.x with Metal Performance Shaders support

## Install PyTorch with Metal support

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements-dev.txt
```

## Verify Metal availability

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.arange(8, device="mps", dtype=torch.float32)
    print("Smoke test:", (x.square().sum()).item())
PY
```

## Run the API on Metal

```bash
export OCTO_DEVICE=mps
python api.py
```

Check the selected backend:

```bash
curl -s http://localhost:8002/health | python -m json.tool
```

## Performance validation

Run the built-in benchmark helper:

```bash
python -m gpu_support --runs 25 --tokens 128
```

Expected outcome on Apple Silicon:

| Metric | CPU baseline | Metal target |
| --- | ---: | ---: |
| Inference latency | 40-80 ms | 8-16 ms |
| Throughput | 1x | 5-10x |
| Memory profile | Stable | Stable, lower host pressure |

## Troubleshooting

### `MPS backend is unavailable on this machine`

- Confirm you are on macOS 12.3+
- Confirm you are using Apple Silicon
- Reinstall a PyTorch build that includes MPS support

### `Metal smoke test failed`

- Close other GPU-heavy applications
- Restart the Python process to clear Metal allocations
- rerun `python - <<'PY' ...` smoke validation above

### API falls back to CPU

`GET /health` exposes `device_fallback_reason`. Common causes:

- missing checkpoint
- unsupported PyTorch build
- runtime Metal initialization failure

## FAQ

**Does this support Intel Macs?**  
No. Metal acceleration is intended for Apple Silicon.

**How do I force CPU for comparison tests?**  
Set `OCTO_DEVICE=cpu`.

**How do I collect a benchmark table for a PR?**  
Run `python -m gpu_support --runs 25 --tokens 128` and paste the markdown table into the PR description.
