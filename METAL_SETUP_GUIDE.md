# macOS Metal Setup Guide — OctoTetrahedral AGI

Get the OctoTetrahedral AGI inference API running with Metal GPU acceleration
on Apple Silicon (M1 / M2 / M3).

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Mac Chip** | Apple M1 | M2 Pro / M3 |
| **macOS** | 12.3 Monterey | 13.x Ventura or later |
| **Python** | 3.9 | 3.11 |
| **PyTorch** | 2.0 | 2.2+ |
| **Unified Memory** | 8 GB | 16 GB |
| **Disk** | 3 GB | 5 GB |

> **Note**: Intel Macs do not have the MPS (Metal Performance Shaders) backend.
> This guide applies only to Apple Silicon Macs (M-series chips).

---

## Installation

### 1. Install Python

```bash
# Using Homebrew (recommended)
brew install python@3.11

# Verify
python3 --version   # Python 3.11.x
```

### 2. Create a Virtual Environment

```bash
cd octotetrahedral-agi
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with Metal/MPS Support

```bash
# PyTorch 2.x includes MPS support — no special installation needed
pip install torch torchvision

# Verify MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expected output: MPS available: True
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

---

## Enabling Metal Acceleration

### Option A: Environment Variable (Recommended)

```bash
export OCTO_DEVICE=mps
python api.py
```

### Option B: Automatic Detection

The API auto-detects Metal and uses it without any configuration:

```bash
# Just run the API — Metal is detected automatically
python api.py
```

Expected startup log:

```
INFO:     🍎 Metal (MPS) available — using Apple Silicon GPU
INFO:     ✅ Model loaded on mps
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8002
```

### Verify Metal is Active

```bash
curl http://localhost:8002/health
```

Expected response:

```json
{
  "status": "healthy",
  "model": "OctoTetrahedralModel",
  "device": "mps",
  "device_type": "mps",
  "accelerator": "mps"
}
```

---

## Verification Script

Run this to confirm your Metal setup is working end-to-end:

```python
import torch

# 1. Check MPS availability
assert torch.backends.mps.is_available(), "MPS not available"
print("✅ MPS backend available")

# 2. Smoke test: create tensors on Metal
a = torch.tensor([1.0, 2.0, 3.0], device="mps")
b = torch.tensor([4.0, 5.0, 6.0], device="mps")
c = (a + b).cpu()
assert c.tolist() == [5.0, 7.0, 9.0]
print("✅ Metal tensor ops working")

# 3. Embedding on Metal
emb = torch.nn.Embedding(1000, 128).to("mps")
ids = torch.tensor([1, 2, 3], device="mps")
out = emb(ids)
assert out.shape == (3, 128)
print("✅ Metal embedding working")

# 4. Report device info
print(f"\nDevice: mps (Metal Performance Shaders)")
print(f"Memory: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB allocated")
print("\n🍎 Apple Silicon Metal acceleration ready!")
```

---

## Performance Expectations

| Device | Mean Latency | Throughput | vs CPU |
|--------|-------------|-----------|--------|
| CPU (M2 Pro) | ~45 ms | ~22 req/s | 1× |
| Metal/MPS (M2 Pro) | ~6–8 ms | ~130+ req/s | **~6×** |
| CPU (cloud server) | ~65 ms | ~16 req/s | 1× |

> Performance varies by chip generation. M3 chips typically outperform M2 by 20–30%.

---

## Troubleshooting

### MPS not available after installing PyTorch

```
python -c "import torch; print(torch.backends.mps.is_available())"
# False
```

**Fixes**:
1. Verify macOS ≥ 12.3: `sw_vers -productVersion`
2. Upgrade PyTorch: `pip install --upgrade torch`
3. Verify you're on Apple Silicon: `uname -m` should print `arm64`

### Metal inference crashes / hangs

If the API starts with Metal but hangs on the first request:

```bash
# Force CPU fallback
export OCTO_DEVICE=cpu
python api.py
```

The API also auto-detects Metal failures and falls back to CPU transparently.

Known issue: some complex model architectures with large weight matrices can
trigger MPS memory pressure on M1 chips with 8 GB memory. Upgrade to 16 GB
or use the CPU fallback.

### Out of memory on Metal

```
RuntimeError: MPS backend out of memory
```

**Fixes**:
1. Close other GPU-intensive apps (Final Cut Pro, Blender, etc.)
2. Reduce input batch size (max is already 256 tokens)
3. Add `torch.mps.empty_cache()` between requests
4. Use `export OCTO_DEVICE=cpu` as fallback

### Slow first request

Expected behavior. Metal JIT-compiles shader code on first use. Subsequent
requests are at full speed. To pre-warm:

```bash
# After starting the server, send one warm-up request:
curl -X POST http://localhost:8002/predict \
     -H "Content-Type: application/json" \
     -d '{"input_ids": [1]}'
```

---

## Running the Live Benchmark on macOS

```bash
# Auto-detect and compare CPU vs Metal
python scripts/benchmark_live.py

# Export results
python scripts/benchmark_live.py --export md --output /tmp/metal_benchmark.md
cat /tmp/metal_benchmark.md
```

Example output on M2 Pro:

```
------------------------------------------------------------
 OctoTetrahedral AGI — Live Benchmark Results
 Generated: 2026-07-23 19:00:00 UTC
------------------------------------------------------------
Device          Mean (ms)  Min    p50    p95    p99    Max    Req/s  Speedup
------------------------------------------------------------
CPU                 44.20  42.10  43.80  46.50  48.10  52.00   22.6  baseline
Metal (MPS)          6.30   5.80   6.20   7.40   8.10   9.50  158.7    7.0×
------------------------------------------------------------
```

---

## Additional Resources

- [PyTorch MPS documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
