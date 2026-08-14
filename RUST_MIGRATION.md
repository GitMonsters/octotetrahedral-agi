# Rust Migration Guide — Incremental Rustification

This document captures the **strategy and rationale** for the Rust work in this
repository.  The goal is **not** a full rewrite of the Python codebase. Instead,
we follow an incremental approach: Python remains the canonical layer for
orchestration, solver experimentation, and rapid iteration, while Rust owns
the performance-sensitive, determinism-critical, and safety-critical subsystems.

---

## Guiding principle

> Rewrite to Rust **only** where profiling proves benefit or where safety /
> determinism matter.  Grow the Rust surface area gradually — never rewrite
> what Python already does well.

---

## Layer responsibilities

| Layer | Language | Rationale |
|-------|----------|-----------|
| Solver experimentation (ARC, re-ARC, DSL) | **Python** | Rapid iteration; algorithmic, not runtime-bound |
| Research orchestration (`workflow.py`, `train_*.py`) | **Python** | Stability of interface matters more than speed |
| Model architecture prototyping | **Python** | PyTorch ecosystem; too expensive to port |
| **Core forward pass / hub-sync / tensor decomposition** | **Rust** (`src/model`) | Hot path; pure math; determinism required |
| **Inference service / request–response pipeline** | **Rust** (`src/inference`) | Throughput-sensitive; concurrency-safe pool |
| **HTTP serving** | **Rust** (`src/serve`) | Already implemented in `octo-parallel-rs`; Axum is lower overhead than FastAPI for this use case |
| **Monitoring / metrics** | **Rust** (`src/monitoring`) | Lock-free ring buffer; no GIL contention |
| **Deterministic eval harness** | **Rust** (`src/eval`) | Reproducibility across platforms |
| **Config / checkpoint serialisation** | **Rust** (`src/adaptation`) | Stable wire format; no Python import required |

---

## Integration boundary between Python and Rust

### Option A — HTTP (current, zero-dependency)

The `octo-serve` binary exposes the inference service on `localhost:8000`.
Python calls it via `requests` / `httpx`:

```python
import requests

resp = requests.post("http://localhost:8000/infer", json={
    "request_id": "abc",
    "limb_states": [0.5] * 8,
    "task_signal": "reasoning",
})
data = resp.json()
```

This is already the pattern used by `octo-parallel-rs` and the Python
`ngvt_http_real_integration.py` module.

### Option B — PyO3 FFI (future, lower latency)

Add `pyo3` to `Cargo.toml` as an optional feature and expose a
`#[pymodule]` that wraps `InferenceService.infer()`.  This eliminates the
HTTP round-trip for latency-critical callers.

```toml
[features]
python-bindings = ["pyo3/extension-module"]

[dependencies]
pyo3 = { version = "0.22", optional = true }
```

```rust
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pyfunction]
fn infer_py(limb_states: Vec<f32>, task_signal: Option<String>) -> PyResult<String> {
    // ... call InferenceService::infer, return JSON string
}
```

The existing `rustyworm_bridge.py` in this repo shows the intended pattern for
PyO3 bridge modules.

---

## What should **stay in Python**

- `arc_*.py`, `re_arc_*.py` — ARC solver logic
- `train_*.py` — training loops (PyTorch-native)
- `workflow.py` — orchestration entry point
- `serve.py` — FastAPI server (kept for Python model parity testing)
- `ngvt_*.py` — NGVT experiment harness
- All notebooks and research scripts

---

## What should **move to Rust first**

Priority order based on impact / stability:

1. **Forward pass / hub-sync** — already done (`src/model`)
2. **Inference service / batching** — already done (`src/inference`)
3. **HTTP API** — already done (`src/serve`, and existing `octo-parallel-rs`)
4. **Monitoring** — already done (`src/monitoring`)
5. **Block AttnRes** — already done in `octo-parallel-rs/src/block_attn_res.rs`
6. **Eval harness** — already done (`src/eval`)
7. **Checkpoint I/O** — already done (`src/adaptation/checkpoint.rs`)

---

## What should **never be rewritten**

- ARC puzzle solvers — algorithmic bottlenecks require Python-level flexibility
- Training loops — PyTorch autograd is irreplaceable without a Rust tensor backend
- Research notebooks — not worth porting
- DSL synthesiser — highly iterative; Python speed of iteration wins

---

## Decision rule for future Rust work

Move a subsystem to Rust **only** if:

1. A profiler confirms it is a hot path, **or**
2. It requires memory-safety / determinism guarantees, **or**
3. It has a stable, well-tested interface and Python is the bottleneck

---

## Running the Rust components

```bash
# Health check (runs 5 self-tests)
cargo run --bin octo-health

# Single forward pass
cargo run --bin octo-infer -- --limb-states 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 --task-signal reasoning

# Evaluation harness (20 tasks, seed 42)
cargo run --bin octo-eval -- --num-tasks 20 --seed 42

# HTTP inference server (default port 8000)
cargo run --bin octo-serve -- --port 8000

# All tests
cargo test
```

---

## Crate layout

```
src/
├── lib.rs                  # Crate root — re-exports all modules
├── model/
│   ├── mod.rs              # Module root; re-exports OctoModel, ForwardResult
│   ├── geometry.rs         # Tetrahedral geometry (64-point structure)
│   ├── attention.rs        # Geometry-masked self-attention
│   ├── limb.rs             # LimbKind enum + Limb type
│   └── orchestrator.rs     # OctoModel — forward pass, hub-sync, adaptation
├── inference/
│   ├── mod.rs
│   ├── types.rs            # InferenceRequest / Response / Batch types
│   └── service.rs          # InferenceService (model pool, monitor)
├── serve/
│   ├── mod.rs
│   ├── router.rs           # Axum router + AppState
│   └── handlers.rs         # /health /infer /batch /metrics
├── monitoring/
│   ├── mod.rs
│   ├── metrics.rs          # InferenceMonitor (ring-buffer latency stats)
│   └── health.rs           # run_health_check → HealthStatus
├── eval/
│   ├── mod.rs
│   ├── generator.rs        # Deterministic task generator (seeded)
│   ├── runner.rs           # run_eval (drives service with generated tasks)
│   └── scorer.rs           # score_result + EvalSummary
├── adaptation/
│   ├── mod.rs
│   ├── config.rs           # AppConfig (env-var overrides) + ModelAdaptation
│   └── checkpoint.rs       # Checkpoint serialisation (JSON round-trip)
└── bin/
    ├── health.rs           # octo-health CLI
    ├── infer.rs            # octo-infer CLI
    ├── eval.rs             # octo-eval CLI
    └── serve.rs            # octo-serve HTTP server
```
