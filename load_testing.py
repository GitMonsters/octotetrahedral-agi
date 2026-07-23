"""Load testing scenario definitions and reporting helpers for the inference API."""

from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class LoadScenario:
    name: str
    kind: str
    concurrency: int | None = None
    target_rps: int | None = None
    duration_seconds: int = 0


@dataclass(frozen=True)
class LoadMetrics:
    latency_samples_ms: list[float]
    throughput_rps: float
    error_rate: float
    memory_usage_mb: float
    successful_requests: int
    total_requests: int


def required_load_scenarios() -> list[LoadScenario]:
    """Return the required concurrent, sustained, burst, and soak scenarios."""
    scenarios = [
        LoadScenario(name=f"concurrent-{level}", kind="concurrent", concurrency=level)
        for level in (10, 50, 100, 500)
    ]
    scenarios.extend(
        LoadScenario(
            name=f"sustained-{rate}-rps",
            kind="sustained",
            target_rps=rate,
            duration_seconds=60,
        )
        for rate in (30, 50, 100)
    )
    scenarios.append(
        LoadScenario(name="burst-1000-rps", kind="burst", target_rps=1000, duration_seconds=10)
    )
    scenarios.append(
        LoadScenario(name="long-running-10m", kind="long-running", duration_seconds=600)
    )
    return scenarios


def summarize_latencies(latency_samples_ms: list[float]) -> dict[str, float]:
    """Compute latency distribution statistics for a load run."""
    if not latency_samples_ms:
        raise ValueError("latency_samples_ms must not be empty")

    ordered = sorted(latency_samples_ms)
    return {
        "avg_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stddev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p50_ms": ordered[int((len(ordered) - 1) * 0.50)],
        "p95_ms": ordered[int((len(ordered) - 1) * 0.95)],
        "p99_ms": ordered[int((len(ordered) - 1) * 0.99)],
    }


def validate_acceptance(
    metrics: LoadMetrics,
    *,
    min_throughput_rps: float = 16.0,
    max_error_rate: float = 0.01,
    max_p95_latency_ms: float = 250.0,
) -> dict[str, bool]:
    """Validate baseline acceptance criteria for a load scenario."""
    latency = summarize_latencies(metrics.latency_samples_ms)
    result = {
        "throughput_ok": metrics.throughput_rps >= min_throughput_rps,
        "error_rate_ok": metrics.error_rate <= max_error_rate,
        "latency_ok": latency["p95_ms"] <= max_p95_latency_ms,
    }
    result["accepted"] = all(result.values())
    return result


def scaling_recommendations(results: dict[str, LoadMetrics]) -> list[str]:
    """Generate scaling recommendations from load-test results."""
    recommendations: list[str] = []

    if any(metrics.error_rate > 0.01 for metrics in results.values()):
        recommendations.append("Add admission control and backpressure before 500+ concurrency.")

    if any(summarize_latencies(metrics.latency_samples_ms)["p95_ms"] > 250.0 for metrics in results.values()):
        recommendations.append("Enable GPU acceleration to reduce high-percentile latency under load.")

    if any(metrics.memory_usage_mb > 1024 for metrics in results.values()):
        recommendations.append("Tune batch sizes and memory pooling to cap memory growth during sustained tests.")

    if not recommendations:
        recommendations.append("Current baseline is stable; next step is a canary rollout with GPU plus response caching.")

    return recommendations


def generate_load_test_report(results: dict[str, LoadMetrics]) -> str:
    """Create a compact markdown report for the configured load scenarios."""
    lines = [
        "# Load Test Report",
        "",
        "| Scenario | Avg Latency (ms) | Throughput (req/s) | Error Rate | Memory (MB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, metrics in results.items():
        latency = summarize_latencies(metrics.latency_samples_ms)
        lines.append(
            f"| {name} | {latency['avg_ms']:.2f} | {metrics.throughput_rps:.2f} | {metrics.error_rate:.2%} | {metrics.memory_usage_mb:.1f} |"
        )

    lines.extend(["", "## Recommendations for scaling", ""])
    for recommendation in scaling_recommendations(results):
        lines.append(f"- {recommendation}")

    return "\n".join(lines)
