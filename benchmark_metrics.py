"""Benchmark metrics: aggregation, statistics, CSV/JSON export, and rankings.

Computes for each model:
  - Latency statistics (mean, median, p95, p99, stdev) in milliseconds
  - Throughput (requests per second)
  - Token generation speed (tokens per second)
  - Memory usage (MB, RSS delta during benchmark)
  - Cost per 1M tokens (USD)
  - Accuracy (from reasoning / few-shot scenarios)
  - Energy consumption (estimated Wh per 1K tokens)
  - Efficiency score (accuracy / cost trade-off)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import resource
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def peak_memory_mb() -> float:
    """Return current peak RSS in megabytes (Unix only; 0.0 elsewhere)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        if os.uname().sysname == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Token-speed estimation
# ---------------------------------------------------------------------------

def estimate_tokens_per_second(latency_ms: float, n_output_tokens: int = 100) -> float:
    """Estimate token generation speed given mean latency and output length."""
    if latency_ms <= 0:
        return 0.0
    return n_output_tokens / (latency_ms / 1000.0)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sd = sorted(data)
    idx = (pct / 100.0) * (len(sd) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sd) - 1)
    frac = idx - lo
    return sd[lo] * (1 - frac) + sd[hi] * frac


def aggregate_scenario_results(scenario_results: dict[str, Any]) -> dict[str, Any]:
    """Collect latencies and accuracies across all scenarios into summary stats."""
    latencies: list[float] = []
    accuracies: list[float] = []
    tps_values: list[float] = []

    for name, data in scenario_results.items():
        if not isinstance(data, dict) or "error" in data:
            continue
        # Collect raw latency arrays
        if "latencies_ms" in data:
            latencies.extend([v for v in data["latencies_ms"] if isinstance(v, (int, float))])
        elif "mean_ms" in data:
            latencies.append(float(data["mean_ms"]))
        elif "mean_latency_ms" in data:
            latencies.append(float(data["mean_latency_ms"]))
        # Collect accuracy
        if "accuracy" in data and isinstance(data["accuracy"], (int, float)):
            accuracies.append(float(data["accuracy"]))
        # Collect throughput from batch scenario
        for batch in data.get("batches", {}).values():
            if "requests_per_sec" in batch:
                tps_values.append(float(batch["requests_per_sec"]))

    return {
        "latency": {
            "mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_ms": _percentile(latencies, 95) if latencies else 0.0,
            "p99_ms": _percentile(latencies, 99) if latencies else 0.0,
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        },
        "accuracy": {
            "mean": statistics.mean(accuracies) if accuracies else 0.0,
            "values": accuracies,
        },
        "throughput_rps": statistics.mean(tps_values) if tps_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Full per-model metrics record
# ---------------------------------------------------------------------------

def compute_model_metrics(
    model: str,
    scenario_results: dict[str, Any],
    cost_per_1m: float,
    energy_wh_per_1k_tokens: float,
    peak_mem_mb: float = 0.0,
    n_output_tokens: int = 100,
) -> dict[str, Any]:
    """Produce a complete metrics record for one model."""
    agg = aggregate_scenario_results(scenario_results)
    mean_latency = agg["latency"]["mean_ms"]
    tokens_per_sec = estimate_tokens_per_second(mean_latency, n_output_tokens)

    # Throughput: prefer batch scenario value; fall back to latency-derived estimate
    throughput_rps = agg["throughput_rps"]
    if throughput_rps == 0.0 and mean_latency > 0:
        throughput_rps = 1000.0 / mean_latency

    accuracy = agg["accuracy"]["mean"]
    # Efficiency: accuracy points per dollar spent per 1M tokens
    efficiency_score = accuracy / (cost_per_1m + 0.01)

    return {
        "model": model,
        "latency_ms": mean_latency,
        "latency_p95_ms": agg["latency"]["p95_ms"],
        "latency_p99_ms": agg["latency"]["p99_ms"],
        "throughput_rps": throughput_rps,
        "tokens_per_sec": tokens_per_sec,
        "memory_mb": peak_mem_mb,
        "cost_per_1m_tokens_usd": cost_per_1m,
        "accuracy": accuracy,
        "energy_wh_per_1k_tokens": energy_wh_per_1k_tokens,
        "efficiency_score": efficiency_score,
        "scenarios": scenario_results,
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_COLUMNS: list[str] = [
    "model",
    "latency_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_rps",
    "tokens_per_sec",
    "memory_mb",
    "cost_per_1m_tokens_usd",
    "accuracy",
    "energy_wh_per_1k_tokens",
    "efficiency_score",
]


def to_csv(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    """Serialise model metrics to a CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for model_metrics in metrics_by_model.values():
        row = {k: model_metrics.get(k, "") for k in _CSV_COLUMNS}
        writer.writerow(row)
    return buf.getvalue()


def save_csv(metrics_by_model: dict[str, dict[str, Any]], path: Path | str) -> None:
    """Write metrics to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_csv(metrics_by_model))
    logger.info("CSV saved → %s", path)


def save_json(data: Any, path: Path | str) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("JSON saved → %s", path)


# ---------------------------------------------------------------------------
# Ranking and summary helpers
# ---------------------------------------------------------------------------

def rank_models(
    metrics_by_model: dict[str, dict[str, Any]],
    key: str = "latency_ms",
    ascending: bool = True,
) -> list[tuple[str, float]]:
    """Return (model, value) pairs sorted by the given metric."""
    sentinel = float("inf") if ascending else 0.0
    items = [
        (m, float(d.get(key, sentinel)))
        for m, d in metrics_by_model.items()
    ]
    return sorted(items, key=lambda x: x[1], reverse=not ascending)


def summary_table(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    """Return a plain-text aligned summary table for logging."""
    cols = [
        ("Model", "model", 24, "s"),
        ("Latency ms", "latency_ms", 12, ".1f"),
        ("Tput rps", "throughput_rps", 10, ".2f"),
        ("Accuracy", "accuracy", 10, ".3f"),
        ("Cost/1M $", "cost_per_1m_tokens_usd", 12, ".4f"),
    ]
    header = "  ".join(f"{label:<{w}}" for label, _, w, _ in cols)
    sep = "-" * len(header)
    rows = [header, sep]
    for model, m in metrics_by_model.items():
        parts: list[str] = []
        for label, key, w, fmt in cols:
            if key == "model":
                parts.append(f"{model:<{w}}")
            else:
                val = m.get(key, 0.0)
                parts.append(f"{val:{w}{fmt}}")
        rows.append("  ".join(parts))
    return "\n".join(rows)
