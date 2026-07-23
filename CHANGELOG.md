# Changelog

All notable changes to OctoTetrahedral AGI are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-07-23

### Added

#### API & Validation
- `api.py`: Input validation for `/predict` endpoint
  - Empty `input_ids` → `400 Bad Request`
  - `input_ids` > 256 tokens → `413 Payload Too Large`
  - Token IDs outside `[0, 50000]` → `400 Bad Request`
  - Non-integer / boolean token IDs → `400 Bad Request`
- `api.py`: Metal/MPS graceful fallback — retries on CPU if Metal inference fails
- `api.py`: Enhanced `/health` endpoint with `device_type` and `accelerator` fields
- `api.py`: Secure checkpoint loading (`weights_only=True`)

#### GPU / Metal Support
- `gpu_support.py`: Device auto-detection (CUDA → MPS → CPU)
- `gpu_support.py`: Smoke tests for CUDA and MPS before selection
- `gpu_support.py`: `resolve_device()` — returns device info dict
- `gpu_support.py`: `clear_device_cache()` — frees GPU cache memory
- `gpu_support.py`: `benchmark_device()` — single-device timing stats
- `gpu_support.py`: `benchmark_comparison_table()` — Markdown comparison table

#### Tests
- `tests/test_api.py`: 11 API endpoint tests covering valid and invalid inputs
- `tests/test_gpu_support.py`: 8 GPU support tests covering device selection

#### Documentation
- `PERFORMANCE_COMPARISON.md`: CPU vs GPU benchmark analysis
- `DEPLOYMENT.md`: Comprehensive deployment guide (Docker, cloud, Kubernetes)
- `BENCHMARK_REPORT.md`: Baseline performance metrics and methodology
- `API_DOCUMENTATION.md`: Full API reference with examples
- `METAL_SETUP_GUIDE.md`: macOS Apple Silicon Metal setup guide
- `RELEASE_NOTES.md`: v1.0.0 release notes
- `CHANGELOG.md`: This file
- `PROJECT_SUMMARY.md`: Project overview and roadmap

#### Tooling
- `scripts/benchmark_live.py`: Live benchmark with JSON/CSV/Markdown export

### Changed
- `api.py`: Device selection now uses `gpu_support.resolve_device()` (replaces
  inline `torch.cuda.is_available()` check)
- `api.py`: Removed unused `import json`

### Fixed
- `api.py`: `500 Internal Server Error` on empty `input_ids` → now `400`
- `api.py`: `500 Internal Server Error` on oversized batches → now `413`
- `api.py`: `500 Internal Server Error` on out-of-range tokens → now `400`
- `api.py`: Insecure `torch.load(..., weights_only=False)` → `weights_only=True`

### Performance
- CPU baseline: 65.29 ms mean latency, 16.20 req/s throughput
- Metal projected: ~8 ms mean latency, 150+ req/s (~8× speedup)
- CUDA projected: ~7.5 ms mean latency, 155+ req/s (~9× speedup)

### Known Issues
- MPS backend has known instability on early M1 chips with complex models;
  the API automatically falls back to CPU in this case.
- `torch.compile()` integration is not enabled by default (adds startup overhead).

### Roadmap (v1.1.0)
- Enable `torch.compile()` for ~15% latency reduction
- Add streaming responses for long-running inference
- Add batch inference endpoint (`/predict/batch`)
- Add OpenAPI schema for token types

---

## [0.9.x] — 2026 (Development)

### Added
- Initial OctoTetrahedral model architecture (8-limb processing)
- ARC-AGI training pipeline (`train_arc.py`)
- Eval harness with 62 test cases (`tests/test_eval_harness.py`)
- Unified stack architecture (`unified/`)
- Cognitive layer (`cognition.py`, `cognitive/`)
- Workflow orchestrator (`workflow.py`)
- Health check endpoint (`health_check.py`)
- Inference service (`inference.py`, `inference_service.py`)
- Monitoring module (`monitoring.py`)
- Configuration system (`config.py`)

---

[1.0.0]: https://github.com/GitMonsters/octotetrahedral-agi/releases/tag/v1.0.0
[0.9.x]: https://github.com/GitMonsters/octotetrahedral-agi/commits/main
