# OctoTetrahedral AGI — Architecture & Compound Workflow

## Overview

This repository implements a multi-limb AGI architecture built around a **unified cognitive stack** (`unified/`).  Three canonical runtime paths expose the model for different use-cases:

| Path | Entrypoint | Purpose |
|------|-----------|---------|
| HTTP inference server | `serve.py` | REST API for text/token generation |
| Python inference API | `inference_service.py` + `workflow.py` | Programmatic forward-pass API with monitoring |
| GPU training | `train_arc.py` | ARC-AGI training loop |

---

## Compound Workflow (`workflow.py`)

`workflow.py` is the **canonical orchestrator** that wires together the major model lifecycle stages:

```
  production_config
        │
        ▼
  InferenceMonitor  ──────────────────────────────────────┐
        │                                                   │
        ▼                                                   │
  InferenceService  ──► infer() ──► UnifiedForwardModel    │
        │                                                   │
        ▼                                                   │
  run_health_check()                                        │
        │                                           monitoring stats
        ▼                                                   │
  eval_harness (optional)  ◄──────────────────────────────┘
        │
        ▼
  serve.py (subprocess, optional)
```

### Modes

| Mode | CLI flag | Description |
|------|----------|-------------|
| Health check | `--mode health-check` | Self-test the inference service and print a diagnostics report |
| Single inference | `--mode inference` | Run one forward pass and print the result |
| Evaluate | `--mode evaluate` | Run the eval-harness benchmark against the live service |
| Serve | `--mode serve` | Health-check, then launch `serve.py` as a subprocess |

### Quick start

```bash
# Self-test — confirms the model and pipeline are healthy
python workflow.py --mode health-check

# Single inference
python workflow.py --mode inference \
    --limb-states 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    --task-signal reasoning

# Evaluation benchmark (20 tasks, deterministic)
python workflow.py --mode evaluate --num-tasks 20 --eval-seed 42

# HTTP serving (after a health check)
python workflow.py --mode serve -- --scale tiny --port 8080
```

### Programmatic usage

```python
from workflow import CompoundWorkflow

# Context-manager (recommended)
with CompoundWorkflow() as wf:
    status = wf.health_check()           # Stage 3
    result = wf.infer([0.5]*8)           # Stage 2
    summary = wf.evaluate(num_tasks=20)  # Stage 6

# Or manually
wf = CompoundWorkflow()
wf.initialize()          # Stages 1 & 4: config, service, monitor
status = wf.health_check()
wf.shutdown()
```

---

## Module Map

### Primary runtime modules

| Module | Role |
|--------|------|
| `workflow.py` | **Compound orchestrator** — wires all lifecycle stages |
| `inference_service.py` | Connection-pooled forward-pass service with retry/timeout |
| `health_check.py` | Self-test suite and diagnostics report |
| `monitoring/` | `InferenceMonitor`, `MetricsRecorder`, `MonitoringSystem`, CLI/web dashboards |
| `serve.py` | FastAPI HTTP server for text/token generation |
| `train_arc.py` | GPU training entrypoint (ARC-AGI dataset) |
| `eval_harness/` | Deterministic benchmark: task gen, scoring, regression tracking |

### Supporting modules

| Module | Role |
|--------|------|
| `production_config.py` | All tunable parameters and env-specific defaults |
| `api_types.py` | TypedDict request/response types + JSON serialization |
| `unified/forward_model.py` | UnifiedForwardModel — the canonical forward-pass implementation |
| `unified/feedback_loop.py` | Bidirectional limb coupling |

### Experimental / archival areas

The following areas exist in the repo but are **not part of the primary compound workflow**.  Treat them as research artefacts:

- `ngvt_*.py` — NGVT multi-server orchestration prototypes
- `arc_*.py`, `eval_arc_*.py` — ARC solver experiments
- `cognition.py`, `cognitive/`, `cognitive_layer.py` — early cognitive stack prototypes
- `octo_server.py`, `ngvt_ultra_simple_server.py` — alternate server experiments

---

## Monitoring namespace

The `monitoring` symbol resolves to the **`monitoring/` package** (Python always prefers a package over a same-named `.py` file).  The `monitoring/` package exports all symbols needed by the inference pipeline:

```python
from monitoring import InferenceMonitor, CoherenceAlert, MonitoringStats  # ← package
from monitoring import MetricsRecorder, MonitoringSystem, MonitoringConfig
```

The canonical implementation of `InferenceMonitor` lives in `monitoring/inference_monitor.py`.
The top-level `monitoring.py` file is a backward-compatibility stub that re-exports those symbols.

---

## Deployment decision tree

```
Need HTTP REST API?
  └─► python serve.py --scale tiny --port 8080

Need Python inference API?
  └─► from inference_service import InferenceService  (or use workflow.py)

Need to run the full lifecycle (init + health + infer + eval)?
  └─► python workflow.py --mode health-check / inference / evaluate

Need GPU training?
  └─► python train_arc.py --resume checkpoints/arc/arc_step_2500.pt
```

---

## Internal architecture of the unified stack

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details on the quantum-biological
unified cognitive stack (`unified/`), including the RNA adaptation layer, quantum coupling
operators, feedback loop, and state transitions.
