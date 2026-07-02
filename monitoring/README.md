# Monitoring — Real-Time Analytics for the Unified Cognitive Stack

This package provides a three-layer observability stack for `UnifiedForwardModel` inference calls.

| Layer | Module | Description |
|-------|--------|-------------|
| 1 | `metrics_recorder.py` | In-process circular-buffer recorder |
| 2 | `cli_monitor.py` | Terminal TUI live monitor |
| 3 | `web_dashboard.py` | FastAPI web dashboard + Prometheus export |
| — | `integration.py` | `MonitoringSystem` wires all layers together |
| — | `config.py` | Centralized configuration |

---

## Installation

The recorder has no extra dependencies beyond the standard library.

For the web dashboard and CLI monitor install:

```bash
pip install fastapi uvicorn websockets httpx
```

---

## Basic Usage — In-Process Recorder

```python
from unified.forward_model import UnifiedForwardModel
from monitoring.metrics_recorder import MetricsRecorder

model = UnifiedForwardModel()
recorder = MetricsRecorder()

recorder.start_recording(model)
model.forward([0.1] * 8, task_signal="reasoning")
stats = recorder.get_rolling_stats()
recorder.stop_recording()

print(stats["current"]["coherence"])
print(stats["all"]["latency_p99"])
```

### Context manager

```python
with MetricsRecorder() as recorder:
    recorder.start_recording(model)
    model.forward([0.5] * 8)
    stats = recorder.get_rolling_stats()
```

### Export to CSV

```python
recorder.export_csv("/tmp/inferences.csv")
```

---

## CLI Monitor Tutorial

### Standalone

```bash
python -m monitoring.cli_monitor
```

### Custom thresholds

```bash
python -m monitoring.cli_monitor \
  --coherence-alert 0.85 \
  --latency-alert 30 \
  --update-freq 0.5
```

### Key bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset stats |
| `s` | Toggle detail level |
| `Ctrl-C` | Quit |

---

## Web Dashboard

### Start the server

```bash
python -m monitoring.web_dashboard --port 8000
```

Then open **http://localhost:8000/** in a browser.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live dashboard (HTML) |
| `GET /health` | Service health check |
| `GET /api/metrics/current` | Current stats (JSON) |
| `GET /api/metrics/history?minutes=5` | Time-series data |
| `GET /api/metrics/export?format=prometheus` | Prometheus text |
| `GET /metrics` | Prometheus text (canonical URL) |
| `WS /ws/metrics` | Streaming updates (1 Hz) |

---

## MonitoringSystem — All Layers Together

```python
from unified.forward_model import UnifiedForwardModel
from monitoring.integration import MonitoringSystem

model = UnifiedForwardModel()

with MonitoringSystem(model, enable_cli=False, enable_web=True) as monitor:
    for i in range(100):
        model.forward([0.5] * 8, task_signal="batch_run")
    stats = monitor.get_stats()
    print(f"Avg coherence: {stats['all']['coherence_mean']:.4f}")
```

---

## Prometheus + Grafana Integration

Point Prometheus at `http://localhost:8000/metrics`.

Example `prometheus.yml` scrape config:

```yaml
scrape_configs:
  - job_name: unified_stack
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 5s
```

Metrics exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `unified_coherence` | gauge | Current / mean coherence |
| `unified_latency_ms` | summary | Latency p50 / p99 / p99.9 |
| `unified_limbs_active` | gauge | Active cognitive limbs |
| `unified_inference_count` | counter | Total inferences |
| `unified_throughput_rps` | gauge | Requests per second |

---

## Configuration

All settings are in `MonitoringConfig`:

```python
from monitoring.config import MonitoringConfig

config = MonitoringConfig(
    circular_buffer_size=2000,       # keep last 2000 inferences
    cli_update_frequency_sec=0.5,    # refresh CLI every 500 ms
    cli_coherence_threshold=0.85,    # custom SLA threshold
    web_port=9000,                   # web dashboard port
)
```

---

## Performance Considerations

- **Overhead**: The recorder adds `< 1 ms` per inference (a few microseconds in practice).
- **Memory**: Each inference record is ~500 bytes. A 1000-entry buffer uses ~500 KB.
- **Thread safety**: All buffer access is protected by `threading.Lock`.
- **No unbounded growth**: `deque(maxlen=N)` automatically evicts oldest entries.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
→ Run `pip install fastapi uvicorn`

**CLI monitor shows "Waiting for inferences…"**
→ Make sure you called `recorder.start_recording(model)` before running inferences.

**Prometheus shows all zeros**
→ The recorder buffer is empty. Run some inferences first.
