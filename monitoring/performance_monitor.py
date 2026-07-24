"""Lightweight request-level performance monitor for OctoTetrahedral AGI.

Tracks inference latency, throughput, error rates, and optional memory usage
using a bounded sliding-window buffer so memory stays constant over time.
"""

import threading
import time
from collections import deque
from typing import Deque

__all__ = ["PerformanceMonitor"]

_WINDOW_SIZE = 1000  # max records kept in memory


class PerformanceMonitor:
    """Thread-safe sliding-window request performance tracker.

    Example::

        pm = PerformanceMonitor()
        pm.record(latency_ms=23.4)
        pm.record(latency_ms=5.0, error=True)
        stats = pm.get_stats()
        print(stats["avg_latency_ms"])
    """

    def __init__(self, window: int = _WINDOW_SIZE) -> None:
        self._lock = threading.Lock()
        self._window = window
        # Each record: {"ts": float, "latency_ms": float, "error": bool}
        self._records: Deque[dict] = deque(maxlen=window)
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, latency_ms: float, *, error: bool = False) -> None:
        """Record a single request outcome."""
        with self._lock:
            self._records.append(
                {"ts": time.time(), "latency_ms": latency_ms, "error": error}
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a snapshot of current performance statistics."""
        try:
            import psutil  # optional dependency
            mem = psutil.virtual_memory()
            memory_mb: float | None = round(mem.used / 1024 / 1024, 1)
            memory_pct: float | None = mem.percent
        except ImportError:
            memory_mb = None
            memory_pct = None

        with self._lock:
            records = list(self._records)

        uptime_s = time.time() - self._start_time
        total = len(records)
        errors = sum(1 for r in records if r["error"])
        latencies = [r["latency_ms"] for r in records if not r["error"]]

        # Throughput: requests per second over the last 60 s
        window_start = time.time() - 60.0
        recent = [r for r in records if r["ts"] >= window_start]
        throughput_rps = round(len(recent) / 60.0, 3)

        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        p95_latency = (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
            if len(latencies) >= 20
            else avg_latency
        )
        error_rate = round(errors / total * 100, 2) if total else 0.0

        result: dict = {
            "uptime_seconds": round(uptime_s, 1),
            "total_requests": total,
            "error_count": errors,
            "error_rate_pct": error_rate,
            "throughput_rps": throughput_rps,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
        }
        if memory_mb is not None:
            result["memory_used_mb"] = memory_mb
            result["memory_pct"] = memory_pct
        return result

    def get_prometheus_metrics(self) -> str:
        """Return stats formatted as Prometheus text exposition."""
        stats = self.get_stats()
        lines = [
            "# HELP octo_requests_total Total number of inference requests",
            "# TYPE octo_requests_total counter",
            f"octo_requests_total {stats['total_requests']}",
            "# HELP octo_errors_total Total number of failed requests",
            "# TYPE octo_errors_total counter",
            f"octo_errors_total {stats['error_count']}",
            "# HELP octo_error_rate_pct Request error rate (percent)",
            "# TYPE octo_error_rate_pct gauge",
            f"octo_error_rate_pct {stats['error_rate_pct']}",
            "# HELP octo_throughput_rps Requests per second (60 s window)",
            "# TYPE octo_throughput_rps gauge",
            f"octo_throughput_rps {stats['throughput_rps']}",
            "# HELP octo_latency_avg_ms Average inference latency (ms)",
            "# TYPE octo_latency_avg_ms gauge",
            f"octo_latency_avg_ms {stats['avg_latency_ms']}",
            "# HELP octo_latency_p95_ms 95th-percentile inference latency (ms)",
            "# TYPE octo_latency_p95_ms gauge",
            f"octo_latency_p95_ms {stats['p95_latency_ms']}",
        ]
        if "memory_used_mb" in stats:
            lines += [
                "# HELP octo_memory_used_mb Host memory used (MB)",
                "# TYPE octo_memory_used_mb gauge",
                f"octo_memory_used_mb {stats['memory_used_mb']}",
            ]
        lines.append("")  # trailing newline
        return "\n".join(lines)
