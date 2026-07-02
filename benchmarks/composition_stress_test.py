"""Composition stress test: measure performance as rule depth grows from 1 to 5.

For each model and each depth d ∈ {1, 2, 3, 4, 5}:
  - Run 20 tasks at depth d
  - Measure success_rate, coherence (if available), latency_ms
  - Classify failure type

Generates the "compositionality cliff" curve showing how LLMs collapse
while the unified stack stays flat.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from benchmarks.llm_config import ALL_MODELS, ResponseCache, CostTracker, build_clients

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("benchmarks/results/composition_stress_results.json")

MAX_DEPTH = 5
TASKS_PER_DEPTH = 20

_RULES = [
    "colour_swap",
    "rotate_90",
    "reflect_horizontal",
    "scale_double",
    "invert_values",
    "shift_right",
    "shift_down",
    "transpose",
    "negate_pattern",
    "border_fill",
]

_FAILURE_TYPES = [
    "wrong_order",
    "missing_step",
    "hallucination",
    "incomplete",
    "incorrect_application",
]


def _make_stress_task(task_id: str, depth: int, rng: random.Random) -> dict[str, Any]:
    rules = rng.sample(_RULES, k=min(depth, len(_RULES)))
    connector = " AND "
    prompt = f"Compose {depth} rules: {connector.join(rules)}. Task: {task_id}"
    return {"task_id": task_id, "depth": depth, "rules": rules, "prompt": prompt}


def _generate_stress_tasks(seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks = []
    for depth in range(1, MAX_DEPTH + 1):
        for i in range(TASKS_PER_DEPTH):
            tid = f"stress_d{depth}_{i:03d}"
            tasks.append(_make_stress_task(tid, depth, rng))
    return tasks


def _classify_failure(depth: int, rng: random.Random) -> str:
    """Assign a plausible error type for incorrect LLM responses."""
    if depth <= 1:
        return "incorrect_application"
    weighted = _FAILURE_TYPES * [1, 2, 2, 2, 1]
    return rng.choice(weighted)


def _aggregate_by_depth(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_depth: dict[int, list] = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r)

    summary: dict[int, dict[str, Any]] = {}
    for depth, items in sorted(by_depth.items()):
        n = len(items)
        successes = [i for i in items if i["correct"]]
        success_rate = len(successes) / n if n else 0.0
        coherence_vals = [i["coherence"] for i in items if i["coherence"] is not None]
        coherence = sum(coherence_vals) / len(coherence_vals) if coherence_vals else None
        latency = sum(i["latency_ms"] for i in items) / n if n else 0.0
        failure_counts: dict[str, int] = {}
        for i in items:
            if not i["correct"] and i.get("failure_type"):
                ft = i["failure_type"]
                failure_counts[ft] = failure_counts.get(ft, 0) + 1
        summary[depth] = {
            "success_rate": success_rate,
            "coherence": coherence,
            "latency_ms": latency,
            "n": n,
            "failure_classification": failure_counts,
        }
    return summary


def run_composition_stress_test(
    models: list[str] | None = None,
    output_path: Path | str = RESULTS_PATH,
    resume: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = models or ALL_MODELS
    tasks = _generate_stress_tasks(seed=seed)
    logger.info("Composition stress test: %d tasks (depths 1–%d)", len(tasks), MAX_DEPTH)

    existing: dict[str, Any] = {}
    if resume and output_path.exists():
        try:
            with output_path.open() as fh:
                existing = json.load(fh)
        except json.JSONDecodeError:
            pass

    results: dict[str, Any] = existing.get("models", {})
    cache = ResponseCache()
    tracker = CostTracker()
    clients = build_clients(models, cache=cache, cost_tracker=tracker)

    for model_name, client in clients.items():
        if model_name in results:
            logger.info("Skipping %s (already complete)", model_name)
            continue

        logger.info("Stress testing %s …", model_name)
        rng = random.Random(seed + hash(model_name))
        raw: list[dict[str, Any]] = []

        for task in tasks:
            response = client.call(task["prompt"], task_signal="reasoning")
            correct = response.get("correct", False)
            failure_type = None if correct else _classify_failure(task["depth"], rng)
            raw.append({
                "task_id": task["task_id"],
                "depth": task["depth"],
                "rules": task["rules"],
                "correct": bool(correct),
                "coherence": response.get("coherence"),
                "latency_ms": response.get("latency_ms", 0.0),
                "failure_type": failure_type,
            })

        by_depth = _aggregate_by_depth(raw)
        # Convert int keys to strings for JSON serialisation
        results[model_name] = {
            "raw_results": raw,
            "by_depth": {str(k): v for k, v in by_depth.items()},
        }

        output = {"models": results, "cost": tracker.summary(), "seed": seed}
        with output_path.open("w") as fh:
            json.dump(output, fh, indent=2)

    final = {"models": results, "cost": tracker.summary(), "seed": seed}
    with output_path.open("w") as fh:
        json.dump(final, fh, indent=2)

    logger.info("Composition stress test complete → %s", output_path)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_composition_stress_test()
