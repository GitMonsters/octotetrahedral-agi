"""Benchmark task scenarios for the production LLM comparison suite.

Scenarios:
  1. single_latency       — single inference latency (n warmup + n_samples calls)
  2. batch_processing     — sequential throughput at batch sizes 10, 100, 1 000
  3. concurrent_requests  — parallel throughput at 10, 50, 100 concurrent workers
  4. long_context         — latency on 8 K and 16 K token prompts
  5. few_shot             — in-context learning accuracy
  6. reasoning            — MMLU + ARC sampled accuracy and latency
"""

from __future__ import annotations

import concurrent.futures
import logging
import statistics
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt bank
# ---------------------------------------------------------------------------

_SINGLE_PROMPT = "What is the capital of France? Answer in one word."

_REASONING_PROMPTS: list[str] = [
    "Solve step by step: If a car travels at 60 km/h for 2.5 hours, how far does it travel?",
    "Is this argument valid? All mammals are animals. All dogs are mammals. Therefore all dogs are animals.",
    "What is 17 × 23 + 44 / 4?",
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "If x + 7 = 15 and y = 2x, what is y?",
]

# Q: 5 + 8 = 13 is the expected answer
_FEW_SHOT_PROMPT = (
    "Q: What is 2 + 2? A: 4\n"
    "Q: What is 3 × 3? A: 9\n"
    "Q: What is 10 - 6? A: 4\n"
    "Q: What is 5 + 8? A:"
)

_MMLU_SAMPLES: list[dict[str, str]] = [
    {
        "prompt": "What is the powerhouse of the cell? A) Nucleus B) Mitochondria C) Ribosome D) Lysosome",
        "answer": "B",
    },
    {
        "prompt": "Which planet is closest to the Sun? A) Venus B) Earth C) Mercury D) Mars",
        "answer": "C",
    },
    {
        "prompt": "What is H2O? A) Salt B) Sugar C) Water D) Oxygen",
        "answer": "C",
    },
    {
        "prompt": "Who wrote Hamlet? A) Dickens B) Shakespeare C) Austen D) Twain",
        "answer": "B",
    },
    {
        "prompt": (
            "What is the approximate speed of light? "
            "A) 3×10^8 m/s B) 3×10^6 m/s C) 3×10^10 m/s D) 300 m/s"
        ),
        "answer": "A",
    },
]

_ARC_SAMPLES: list[dict[str, str]] = [
    {
        "prompt": "Why do plants need sunlight? A) To grow B) To breathe C) To drink water D) To sleep",
        "answer": "A",
    },
    {
        "prompt": (
            "What causes seasons on Earth? "
            "A) Distance from Sun B) Earth's tilt C) Moon phases D) Ocean currents"
        ),
        "answer": "B",
    },
]


# ---------------------------------------------------------------------------
# Helper
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


_FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "


def _make_long_context_prompt(n_tokens: int) -> str:
    """Generate a prompt approximating n_tokens (4 chars ≈ 1 token)."""
    base = "Please summarise the following text and answer: What is the main theme?\n\n"
    suffix = "\n\nMain theme (one sentence):"
    target_chars = n_tokens * 4
    fill_chars = max(0, target_chars - len(base) - len(suffix))
    filler = (_FILLER_SENTENCE * (fill_chars // len(_FILLER_SENTENCE) + 1))[:fill_chars]
    return base + filler + suffix


# ---------------------------------------------------------------------------
# Scenario 1: Single inference latency
# ---------------------------------------------------------------------------

def run_single_latency(
    client: Any,
    n_warmup: int = 2,
    n_samples: int = 10,
) -> dict[str, Any]:
    """Measure single-inference latency across n_samples calls."""
    for _ in range(n_warmup):
        client.call(_SINGLE_PROMPT)

    latencies: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        result = client.call(_SINGLE_PROMPT)
        wall_ms = (time.perf_counter() - t0) * 1000
        latencies.append(result.get("latency_ms", wall_ms))

    return {
        "scenario": "single_latency",
        "n_samples": n_samples,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Scenario 2: Batch processing
# ---------------------------------------------------------------------------

def run_batch_processing(
    client: Any,
    batch_sizes: tuple[int, ...] = (10, 100, 1000),
) -> dict[str, Any]:
    """Measure sequential throughput at different batch sizes."""
    prompts_pool = [f"{_SINGLE_PROMPT} (variant {i})" for i in range(max(batch_sizes))]
    batches: dict[str, Any] = {}

    for bs in batch_sizes:
        prompts = prompts_pool[:bs]
        t0 = time.perf_counter()
        for p in prompts:
            client.call(p)
        elapsed = time.perf_counter() - t0
        batches[str(bs)] = {
            "batch_size": bs,
            "elapsed_s": elapsed,
            "requests_per_sec": bs / elapsed if elapsed > 0 else 0.0,
        }

    return {"scenario": "batch_processing", "batches": batches}


# ---------------------------------------------------------------------------
# Scenario 3: Concurrent requests
# ---------------------------------------------------------------------------

def run_concurrent_requests(
    client: Any,
    concurrency_levels: tuple[int, ...] = (10, 50, 100),
    requests_per_level: int = 20,
) -> dict[str, Any]:
    """Measure throughput and latency under concurrent load."""
    levels: dict[str, Any] = {}

    for concurrency in concurrency_levels:
        prompts = [f"{_SINGLE_PROMPT} (concurrent_{i})" for i in range(requests_per_level)]
        latencies: list[float] = []
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(client.call, p) for p in prompts]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    latencies.append(res.get("latency_ms", 0.0))
                except Exception as exc:
                    logger.warning("Concurrent request failed: %s", exc)

        wall_elapsed = time.perf_counter() - t0
        levels[str(concurrency)] = {
            "concurrency": concurrency,
            "requests": requests_per_level,
            "wall_elapsed_s": wall_elapsed,
            "requests_per_sec": len(latencies) / wall_elapsed if wall_elapsed > 0 else 0.0,
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p99_latency_ms": _percentile(latencies, 99) if latencies else 0.0,
            "errors": requests_per_level - len(latencies),
        }

    return {"scenario": "concurrent_requests", "levels": levels}


# ---------------------------------------------------------------------------
# Scenario 4: Long context handling
# ---------------------------------------------------------------------------

def run_long_context(
    client: Any,
    context_sizes_k: tuple[int, ...] = (8, 16),
    n_samples: int = 3,
) -> dict[str, Any]:
    """Evaluate latency on long prompts (8 K and 16 K tokens)."""
    contexts: dict[str, Any] = {}

    for size_k in context_sizes_k:
        prompt = _make_long_context_prompt(size_k * 1024)
        latencies: list[float] = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            res = client.call(prompt)
            wall_ms = (time.perf_counter() - t0) * 1000
            latencies.append(res.get("latency_ms", wall_ms))

        contexts[f"{size_k}k"] = {
            "context_tokens": size_k * 1024,
            "mean_latency_ms": statistics.mean(latencies),
            "max_latency_ms": max(latencies),
        }

    return {"scenario": "long_context", "contexts": contexts}


# ---------------------------------------------------------------------------
# Scenario 5: Few-shot learning
# ---------------------------------------------------------------------------

def run_few_shot(client: Any, n_samples: int = 5) -> dict[str, Any]:
    """Test few-shot in-context learning (expected answer: 13)."""
    latencies: list[float] = []
    correct = 0

    for _ in range(n_samples):
        t0 = time.perf_counter()
        res = client.call(_FEW_SHOT_PROMPT)
        wall_ms = (time.perf_counter() - t0) * 1000
        latencies.append(res.get("latency_ms", wall_ms))
        if "13" in str(res.get("answer", "")):
            correct += 1

    return {
        "scenario": "few_shot",
        "n_samples": n_samples,
        "accuracy": correct / n_samples if n_samples > 0 else 0.0,
        "mean_latency_ms": statistics.mean(latencies),
    }


# ---------------------------------------------------------------------------
# Scenario 6: Reasoning (MMLU + ARC)
# ---------------------------------------------------------------------------

def run_reasoning(client: Any) -> dict[str, Any]:
    """Run MMLU and ARC sampled accuracy and latency measurement."""
    latencies: list[float] = []
    correct = 0
    total_items = _MMLU_SAMPLES + _ARC_SAMPLES

    for item in total_items:
        t0 = time.perf_counter()
        res = client.call(item["prompt"])
        wall_ms = (time.perf_counter() - t0) * 1000
        latencies.append(res.get("latency_ms", wall_ms))
        if item["answer"].lower() in str(res.get("answer", "")).lower():
            correct += 1

    total = len(total_items)
    return {
        "scenario": "reasoning",
        "total_tasks": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "benchmarks": {"mmlu_n": len(_MMLU_SAMPLES), "arc_n": len(_ARC_SAMPLES)},
    }


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

_ALL_SCENARIO_NAMES = [
    "single_latency",
    "batch_processing",
    "concurrent_requests",
    "long_context",
    "few_shot",
    "reasoning",
]


def run_all_scenarios(
    client: Any,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """Run all (or a specified subset of) scenarios for one model client."""
    dispatch: dict[str, Any] = {
        "single_latency": lambda: run_single_latency(client),
        "batch_processing": lambda: run_batch_processing(client),
        "concurrent_requests": lambda: run_concurrent_requests(client),
        "long_context": lambda: run_long_context(client),
        "few_shot": lambda: run_few_shot(client),
        "reasoning": lambda: run_reasoning(client),
    }

    to_run = {
        k: v for k, v in dispatch.items()
        if scenarios is None or k in scenarios
    }

    results: dict[str, Any] = {}
    for name, fn in to_run.items():
        logger.info("  Running scenario: %s", name)
        try:
            results[name] = fn()
        except Exception as exc:
            logger.error("  Scenario %s failed: %s", name, exc)
            results[name] = {"scenario": name, "error": str(exc)}

    return results
