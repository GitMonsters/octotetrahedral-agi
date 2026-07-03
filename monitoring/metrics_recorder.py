"""Real-time metrics recorder for the unified 8-limb forward model.

Hooks `UnifiedForwardModel.forward()` calls (via `MetricsRecorder.instrument`)
and keeps a memory-efficient rolling window of recent inferences: coherence,
latency, coupling strength, active limb count, and action channel. Consumers
(the CLI monitor, the web dashboard, or any other caller) read a point-in-time
`snapshot()` that is safe to call from another thread while inference keeps
recording.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass

from unified.forward_model import UnifiedForwardModel, UnifiedForwardResult

DEFAULT_WINDOW_SIZE = 100

# A limb is considered "active" when its residual (deviation from the shared
# component) exceeds this magnitude, i.e. it is contributing distinct signal
# rather than merely tracking the mean. This is a display heuristic only and
# does not affect model behavior.
DEFAULT_ACTIVE_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class MetricSample:
    """A single recorded inference event."""

    timestamp: float
    coherence: float
    coupling_strength: float
    latency_ms: float
    action_channel: int
    task_signal: str
    limbs_active: int
    limb_count: int


class MetricsRecorder:
    """Rolling-window recorder of `UnifiedForwardModel` inference metrics."""

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        active_threshold: float = DEFAULT_ACTIVE_THRESHOLD,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer")

        self.window_size = window_size
        self.active_threshold = active_threshold
        self._samples: deque[MetricSample] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._start_time = time.monotonic()

    def record(
        self,
        result: UnifiedForwardResult,
        latency_ms: float,
        task_signal: str | None = None,
    ) -> MetricSample:
        """Record the metrics for one completed `forward()` call."""
        limbs_active = sum(1 for residual in result["residuals"] if abs(residual) > self.active_threshold)
        sample = MetricSample(
            timestamp=time.time(),
            coherence=result["coherence"],
            coupling_strength=result["coupling_strength"],
            latency_ms=latency_ms,
            action_channel=result["action_channel"],
            task_signal=(task_signal or "default").strip().lower(),
            limbs_active=limbs_active,
            limb_count=len(result["limb_states"]),
        )
        with self._lock:
            self._samples.append(sample)
            self._total_requests += 1
        return sample

    def instrument(self, model: UnifiedForwardModel) -> "InstrumentedForwardModel":
        """Wrap `model` so every `forward()` call is transparently timed and recorded."""
        return InstrumentedForwardModel(model, self)

    def history(self) -> list[dict]:
        """Return the current rolling window as a list of plain dicts (oldest first)."""
        with self._lock:
            samples = list(self._samples)
        return [
            {
                "timestamp": sample.timestamp,
                "coherence": sample.coherence,
                "coupling_strength": sample.coupling_strength,
                "latency_ms": sample.latency_ms,
                "action_channel": sample.action_channel,
                "task_signal": sample.task_signal,
                "limbs_active": sample.limbs_active,
                "limb_count": sample.limb_count,
            }
            for sample in samples
        ]

    def snapshot(self) -> dict:
        """Return a point-in-time summary of the rolling window.

        Safe to call concurrently with `record()` from another thread.
        """
        with self._lock:
            samples = list(self._samples)
            total_requests = self._total_requests

        if not samples:
            return {
                "sample_count": 0,
                "total_requests": total_requests,
                "requests_per_second": 0.0,
                "coherence_latest": None,
                "coherence_previous": None,
                "coherence_mean": None,
                "coupling_latest": None,
                "latency_latest_ms": None,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "latency_p99_ms": None,
                "limbs_active": None,
                "limb_count": None,
                "action_channel": None,
                "task_signal": None,
            }

        latencies = sorted(sample.latency_ms for sample in samples)
        coherences = [sample.coherence for sample in samples]
        latest = samples[-1]
        previous_coherence = samples[-2].coherence if len(samples) > 1 else None

        if len(samples) >= 2:
            span_seconds = samples[-1].timestamp - samples[0].timestamp
            requests_per_second = (len(samples) - 1) / span_seconds if span_seconds > 0 else 0.0
        else:
            requests_per_second = 0.0

        return {
            "sample_count": len(samples),
            "total_requests": total_requests,
            "requests_per_second": requests_per_second,
            "coherence_latest": latest.coherence,
            "coherence_previous": previous_coherence,
            "coherence_mean": statistics.mean(coherences),
            "coupling_latest": latest.coupling_strength,
            "latency_latest_ms": latest.latency_ms,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "latency_p99_ms": _percentile(latencies, 0.99),
            "limbs_active": latest.limbs_active,
            "limb_count": latest.limb_count,
            "action_channel": latest.action_channel,
            "task_signal": latest.task_signal,
        }

    def reset(self) -> None:
        """Clear all recorded samples and counters."""
        with self._lock:
            self._samples.clear()
            self._total_requests = 0
            self._start_time = time.monotonic()


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile lookup over an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


class InstrumentedForwardModel:
    """Transparent forward()-timing proxy around a `UnifiedForwardModel`."""

    def __init__(self, model: UnifiedForwardModel, recorder: MetricsRecorder) -> None:
        self._model = model
        self._recorder = recorder

    def forward(self, limb_states: list[float], task_signal: str | None = None) -> UnifiedForwardResult:
        start = time.perf_counter()
        result = self._model.forward(limb_states, task_signal=task_signal)
        latency_ms = (time.perf_counter() - start) * 1000
        self._recorder.record(result, latency_ms, task_signal=task_signal)
        return result

    def __getattr__(self, name):
        # Delegate any other attribute access (e.g. limb_count) to the wrapped model.
        return getattr(self._model, name)
