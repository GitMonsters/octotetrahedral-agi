"""Performance metrics for the unified architecture."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unified.forward_model import LegacyForwardAdapter, UnifiedForwardModel


def _benchmark_callable(callable_obj, samples: int = 200) -> float:
    latencies = []
    for _ in range(samples):
        start = time.perf_counter()
        callable_obj()
        latencies.append(time.perf_counter() - start)
    return statistics.mean(latencies)


def _simulate_modular_pipeline(adapter: LegacyForwardAdapter, input_state: list[float]) -> list[float]:
    """Approximate the old multi-pass modular execution path."""
    stage_one = adapter.run(input_state, task_type="reasoning")
    stage_two = adapter.run(stage_one, task_type="language")
    return adapter.run(stage_two, task_type="spatial")


def run_benchmark(samples: int = 200) -> dict[str, float]:
    model = UnifiedForwardModel()
    adapter = LegacyForwardAdapter()
    input_state = [float(index) / 10.0 for index in range(8)]

    unified_time = _benchmark_callable(lambda: model.forward(input_state, task_signal="reasoning"), samples)
    adapter_time = _benchmark_callable(lambda: adapter.run(input_state, task_type="reasoning"), samples)
    modular_time = _benchmark_callable(lambda: _simulate_modular_pipeline(adapter, input_state), samples)

    efficiency_gain = modular_time / unified_time if unified_time > 0 else 0.0
    return {
        "unified_latency_ms": unified_time * 1000,
        "legacy_adapter_latency_ms": adapter_time * 1000,
        "modular_baseline_latency_ms": modular_time * 1000,
        "efficiency_gain": efficiency_gain,
    }


if __name__ == "__main__":
    results = run_benchmark()
    print(results)
