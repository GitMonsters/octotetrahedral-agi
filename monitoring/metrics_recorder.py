"""In-process metrics recorder for UnifiedForwardModel inference calls."""

from __future__ import annotations

import csv
import statistics
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from monitoring.config import MonitoringConfig

if TYPE_CHECKING:
    from unified.forward_model import UnifiedForwardModel


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile of a sorted list (0–100)."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    lo = int(k)
    hi = lo + 1
    frac = k - lo
    if hi >= len(sorted_data):
        return sorted_data[lo]
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


class MetricsRecorder:
    """Thread-safe circular-buffer recorder that wraps UnifiedForwardModel.forward().

    Usage::

        recorder = MetricsRecorder()
        recorder.start_recording(model)
        model.forward(...)  # automatically captured
        stats = recorder.get_rolling_stats()
        recorder.stop_recording()
    """

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self._config = config or MonitoringConfig()
        self._lock = threading.Lock()
        self._buffer: deque[dict[str, Any]] = deque(
            maxlen=self._config.circular_buffer_size
        )
        self._model: "UnifiedForwardModel | None" = None
        self._original_forward: Any = None
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_recording(self, model: "UnifiedForwardModel") -> None:
        """Attach recorder to *model* by wrapping its forward() method."""
        if self._model is not None:
            raise RuntimeError("Already recording. Call stop_recording() first.")
        self._model = model
        self._original_forward = model.forward

        recorder_ref = self  # avoid circular closure issues

        def _instrumented_forward(
            limb_states: list[float], task_signal: str | None = None
        ):
            t0 = time.perf_counter()
            result = recorder_ref._original_forward(limb_states, task_signal=task_signal)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            recorder_ref._record(result, limb_states, task_signal, latency_ms)
            return result

        model.forward = _instrumented_forward  # type: ignore[method-assign]

    def stop_recording(self) -> None:
        """Detach recorder and restore the original forward() method."""
        if self._model is None:
            return
        self._model.forward = self._original_forward  # type: ignore[method-assign]
        self._model = None
        self._original_forward = None

    def reset(self) -> None:
        """Clear all buffered inferences."""
        with self._lock:
            self._buffer.clear()
            self._start_time = time.time()

    def get_all_inferences(self) -> list[dict[str, Any]]:
        """Return a copy of all buffered inference records."""
        with self._lock:
            return list(self._buffer)

    def get_rolling_stats(self) -> dict[str, Any]:
        """Compute and return current rolling statistics."""
        with self._lock:
            records = list(self._buffer)

        if not records:
            return self._empty_stats()

        now = time.time()
        windows = {
            "1min": now - 60,
            "5min": now - 300,
            "15min": now - 900,
        }

        def _stats_for(subset: list[dict]) -> dict[str, Any]:
            if not subset:
                return {}
            latencies = sorted(r["latency_ms"] for r in subset)
            coherences = sorted(r["coherence"] for r in subset)
            return {
                "count": len(subset),
                "latency_p50": _percentile(latencies, 50),
                "latency_p99": _percentile(latencies, 99),
                "latency_p999": _percentile(latencies, 99.9),
                "latency_mean": statistics.mean(latencies),
                "latency_std": statistics.pstdev(latencies),
                "coherence_mean": statistics.mean(coherences),
                "coherence_std": statistics.pstdev(coherences),
                "coherence_p50": _percentile(coherences, 50),
                "coherence_p99": _percentile(coherences, 99),
            }

        all_stats = _stats_for(records)
        window_stats: dict[str, Any] = {}
        for label, cutoff in windows.items():
            subset = [r for r in records if r["timestamp"] >= cutoff]
            window_stats[label] = _stats_for(subset)

        latest = records[-1]
        elapsed = now - self._start_time
        throughput = len(records) / elapsed if elapsed > 0 else 0.0

        return {
            "total_inferences": len(records),
            "throughput_rps": throughput,
            "current": {
                "coherence": latest["coherence"],
                "latency_ms": latest["latency_ms"],
                "coupling_strength": latest["coupling_strength"],
                "phase": latest["phase"],
                "bias": latest["bias"],
                "limbs_active": latest["limbs_active"],
                "action_channel": latest["action_channel"],
                "task_signal": latest["task_signal"],
            },
            "all": all_stats,
            "windows": window_stats,
        }

    def export_csv(self, path: str) -> None:
        """Export all buffered inferences to a CSV file at *path*."""
        with self._lock:
            records = list(self._buffer)

        if not records:
            return

        fieldnames = list(records[0].keys())
        # Flatten limb_activation_vector for CSV
        has_vector = "limb_activation_vector" in fieldnames
        if has_vector:
            fieldnames = [f for f in fieldnames if f != "limb_activation_vector"]
            first_vec = records[0].get("limb_activation_vector", [])
            for i in range(len(first_vec)):
                fieldnames.append(f"limb_{i}")

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                row = {k: v for k, v in rec.items() if k != "limb_activation_vector"}
                if has_vector:
                    for i, val in enumerate(rec.get("limb_activation_vector", [])):
                        row[f"limb_{i}"] = val
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MetricsRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_recording()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        result: dict[str, Any],
        limb_states: list[float],
        task_signal: str | None,
        latency_ms: float,
    ) -> None:
        limbs_active = sum(1 for v in result["limb_states"] if v > 0.5)
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "coherence": result["coherence"],
            "coupling_strength": result["coupling_strength"],
            "phase": result["phase"],
            "bias": result["bias"],
            "limbs_active": limbs_active,
            "action_channel": result["action_channel"],
            "task_signal": task_signal or "",
            "limb_activation_vector": list(result["limb_states"]),
        }
        with self._lock:
            self._buffer.append(record)

        if self._config.enable_per_inference_logging:
            print(
                f"[MetricsRecorder] coherence={record['coherence']:.4f} "
                f"latency={latency_ms:.2f}ms"
            )

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_inferences": 0,
            "throughput_rps": 0.0,
            "current": {},
            "all": {},
            "windows": {"1min": {}, "5min": {}, "15min": {}},
        }
