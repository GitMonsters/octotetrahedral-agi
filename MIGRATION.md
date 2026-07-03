# Migration Guide: Modular Stack to Unified Cognitive Stack

## What changed

The new unified stack consolidates forward execution into `UnifiedForwardModel` while keeping a legacy adapter for existing integrations.

## New primary API

```python
from unified.forward_model import UnifiedForwardModel

model = UnifiedForwardModel()
result = model.forward([0.1] * 8, task_signal="reasoning")
```

Returned dictionary keys:
- `limb_states`
- `shared_component`
- `residuals`
- `coherence`
- `coupling_strength`
- `phase`
- `bias`
- `action_channel`

## Legacy compatibility path

```python
from unified.forward_model import LegacyForwardAdapter

adapter = LegacyForwardAdapter()
limb_states = adapter.run([0.1] * 8, task_type="reasoning")
```

## Recommended rollout

1. Deploy `LegacyForwardAdapter` first to keep existing call signatures stable.
2. Move downstream consumers to `UnifiedForwardModel.forward` result fields.
3. Enable benchmark checks using `python -m benchmarks.unified_perf`.
4. Remove modular wrappers after migration is complete.

## Real-time analytics

The `monitoring` package records live `UnifiedForwardModel.forward()` metrics
(coherence, latency, coupling strength, active limb count, action channel)
into a rolling window:

```python
from monitoring.metrics_recorder import MetricsRecorder
from unified.forward_model import UnifiedForwardModel

model = UnifiedForwardModel()
recorder = MetricsRecorder(window_size=100)
instrumented = recorder.instrument(model)  # transparent forward() proxy

instrumented.forward([0.1] * 8, task_signal="reasoning")
recorder.snapshot()  # {"coherence_latest": ..., "latency_p99_ms": ..., ...}
```

Two ready-to-run viewers sit on top of the recorder:

- `python -m monitoring.cli_monitor` — a `top`-style terminal panel that
  refreshes once a second (`--interval`, `--window`, `--task-signals`,
  `--duration`, `--seed` are all configurable).
- `python -m monitoring.web_dashboard` — a Flask app serving `/` (an
  auto-refreshing HTML dashboard with a coherence sparkline) and
  `/api/metrics` (the JSON snapshot + rolling history), on `--host`/`--port`
  (defaults `127.0.0.1:8765`).

Both viewers drive synthetic demo traffic through the real
`UnifiedForwardModel` by default; wire your own inference calls through
`MetricsRecorder.instrument()` (or call `record()` directly) to observe real
production traffic instead.

Tests: `python -m pytest -q tests/test_metrics_recorder.py tests/test_monitoring_cli.py tests/test_monitoring_dashboard.py`.
