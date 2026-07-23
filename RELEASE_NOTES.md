# Release Notes — OctoTetrahedral AGI

## v1.0.0 — 2026-07-23 🚀

### First Production Release

OctoTetrahedral AGI v1.0.0 is the first stable, production-ready release of the
OctoTetrahedral neural inference API. This release includes comprehensive input
validation, GPU acceleration (CUDA & Metal/MPS), performance benchmarking tools,
and full documentation.

---

### ✨ Major Features

#### Input Validation & Error Handling
- **Empty input rejection** — `POST /predict` with `input_ids: []` returns `400 Bad Request`
  with a human-readable error: `"input_ids must contain at least 1 token."`
- **Oversized batch rejection** — Batches > 256 tokens return `413 Payload Too Large`
- **Token ID range validation** — IDs outside `[0, 50000]` return `400 Bad Request`
- **Type checking** — Non-integer and boolean values rejected with descriptive errors
- **Secure checkpoint loading** — Model checkpoints now loaded with `weights_only=True`

#### GPU Acceleration (CUDA & Metal)
- **Automatic device detection** via `gpu_support.py`:
  1. CUDA (NVIDIA GPU) — if available and smoke-test passes
  2. Metal/MPS (Apple Silicon) — if available and smoke-test passes
  3. CPU fallback — always available
- **MPS graceful fallback** — if Metal inference fails at runtime, the request
  automatically retries on CPU
- **Environment variable overrides**: `OCTO_DEVICE` and `OCTOTETRAHEDRAL_DEVICE`

#### Enhanced Health Endpoint
- `GET /health` now returns device metadata including `device_type` and `accelerator`

#### macOS Metal Setup
- Complete step-by-step [Metal Setup Guide](METAL_SETUP_GUIDE.md) for Apple Silicon
- PyTorch MPS backend configuration and verification instructions
- Performance expectations and troubleshooting for Metal deployments

#### API Documentation
- Full [API Documentation](API_DOCUMENTATION.md) with:
  - Endpoint schemas and examples (cURL, Python, JavaScript)
  - Error reference
  - Performance specifications
  - Best practices

#### Load Testing Suite
- `load_testing.py` — load testing utilities
- `tests/test_load.py` — automated load test scenarios
- Scenarios: concurrent requests, sustained load, burst load

#### Live Benchmark Script
- `scripts/benchmark_live.py` — auto-detects available devices and runs benchmarks
- Exports results as JSON, CSV, or Markdown
- Compares CPU vs Metal vs CUDA performance

#### Comprehensive Documentation
- [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md) — detailed benchmark analysis
- [DEPLOYMENT.md](DEPLOYMENT.md) — deployment guide for all platforms
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — baseline performance metrics
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — project overview and roadmap

---

### 📊 Performance Highlights

| Metric | CPU Baseline | Metal (Apple Silicon) | CUDA (NVIDIA) |
|--------|-------------|----------------------|---------------|
| Avg Latency | 65.29 ms | ~8 ms | ~7.5 ms |
| Throughput | 16.20 req/s | 150+ req/s | 155+ req/s |
| Speedup | 1× | ~8× | ~9× |

---

### 🐛 Bug Fixes

- Fixed `500 Internal Server Error` on empty `input_ids` → now returns proper `400`
- Fixed `500 Internal Server Error` on oversized batches → now returns proper `413`
- Fixed `500 Internal Server Error` on out-of-range token IDs → now returns proper `400`
- Fixed insecure checkpoint loading (`weights_only=False`) → now uses `weights_only=True`

---

### ⚠️ Breaking Changes

None. This is the initial stable release.

---

### 🔄 Migration Guide

If upgrading from a pre-release or development version:

1. **Validation errors** — clients that previously received `500` on invalid inputs
   will now receive `400` or `413`. Update your error-handling logic accordingly.

2. **Health response** — `GET /health` now returns two additional fields:
   `device_type` and `accelerator`. Existing clients that only read `status`,
   `model`, and `device` are unaffected.

3. **Environment variables** — `OCTO_DEVICE` and `OCTOTETRAHEDRAL_DEVICE` are now
   the official way to override device selection.

---

### 📦 Included Assets

- Source code (full repository)
- `DEPLOYMENT.md` — deployment guide
- `PERFORMANCE_COMPARISON.md` — benchmark analysis
- `API_DOCUMENTATION.md` — API reference
- `METAL_SETUP_GUIDE.md` — macOS Metal guide
- `BENCHMARK_REPORT.md` — baseline metrics
- `scripts/benchmark_live.py` — live benchmark tool

---

### 🙏 Acknowledgements

Built with PyTorch, FastAPI, and the ARC-AGI research community.

---

## Pre-release History

### v0.9.x (Development)
- Initial API scaffolding
- OctoTetrahedral model architecture
- ARC-AGI training pipeline
- Evaluation harness (62 test cases)
- Unified stack architecture
- Cognitive layer integration

---

*Full changelog: [CHANGELOG.md](CHANGELOG.md)*
