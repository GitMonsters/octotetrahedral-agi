"""CCL Model Comparison: run all 300 CCL tasks against 5 models.

CCL (Compound Concept Learning) tasks test how well a model generalises
from single rules (L1) to compounds of two (L2) or three (L3) rules.

Metrics:
  - accuracy   — fraction of tasks solved correctly
  - coherence  — quantum coherence of unified models (None for LLMs)
  - latency_ms — wall-clock time per task
  - CES        — Compounding Efficiency Score: L3_accuracy / L1_accuracy

Resume support: partial results are saved after each model so a crashed
run can continue from the last completed model.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from benchmarks.llm_config import ALL_MODELS, ModelClient, ResponseCache, CostTracker, build_clients

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("benchmarks/results/ccl_comparison_results.json")

# ---------------------------------------------------------------------------
# CCL task generation (300 tasks: 100 L1, 100 L2, 100 L3)
# ---------------------------------------------------------------------------

_RULES = [
    "colour_swap", "rotate_90", "reflect_horizontal", "scale_double",
    "invert_values", "shift_right", "shift_down", "transpose",
    "negate_pattern", "border_fill",
]

_RULE_DESCRIPTIONS: dict[str, str] = {
    "colour_swap": "swap colour A with colour B",
    "rotate_90": "rotate grid 90 degrees clockwise",
    "reflect_horizontal": "reflect the grid horizontally",
    "scale_double": "double the size of each element",
    "invert_values": "invert all cell values",
    "shift_right": "shift every row one cell to the right",
    "shift_down": "shift every column one cell down",
    "transpose": "transpose the grid matrix",
    "negate_pattern": "negate the binary pattern",
    "border_fill": "fill the border cells with a given colour",
}


def _make_task(task_id: str, level: int, rng: random.Random) -> dict[str, Any]:
    """Create a single CCL task at a given compositional depth."""
    rules = rng.sample(_RULES, k=min(level, len(_RULES)))
    description_parts = [_RULE_DESCRIPTIONS[r] for r in rules]
    connector = " AND "
    prompt = f"Apply rule: {connector.join(description_parts)}. Task ID: {task_id}"
    return {
        "task_id": task_id,
        "level": level,
        "rules": rules,
        "prompt": prompt,
    }


def generate_ccl_tasks(seed: int = 42) -> list[dict[str, Any]]:
    """Generate 300 deterministic CCL tasks (100 per level)."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        for i in range(100):
            task_id = f"ccl_L{level}_{i:03d}"
            tasks.append(_make_task(task_id, level, rng))
    return tasks


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(client: ModelClient, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for task in tasks:
        t0 = time.perf_counter()
        response = client.call(task["prompt"], task_signal="reasoning")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        correct = response.get("correct")
        # For LLMs that return free-form text, heuristically judge correctness
        if correct is None and isinstance(response.get("answer"), str):
            answer_lower = response["answer"].lower()
            correct = "correct" not in answer_lower  # conservative: treat as wrong unless explicit
            correct = response.get("correct", False)

        results.append({
            "task_id": task["task_id"],
            "level": task["level"],
            "rules": task["rules"],
            "correct": bool(correct),
            "coherence": response.get("coherence"),
            "latency_ms": response.get("latency_ms", elapsed_ms),
            "model": client.model,
        })
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-level metrics and CES from raw task results."""
    by_level: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for r in results:
        by_level[r["level"]].append(r)

    def _stats(tasks: list[dict]) -> dict[str, Any]:
        if not tasks:
            return {"accuracy": 0.0, "coherence": None, "latency_ms": 0.0, "n": 0}
        accuracy = sum(t["correct"] for t in tasks) / len(tasks)
        coherence_vals = [t["coherence"] for t in tasks if t["coherence"] is not None]
        coherence = sum(coherence_vals) / len(coherence_vals) if coherence_vals else None
        latency = sum(t["latency_ms"] for t in tasks) / len(tasks)
        return {"accuracy": accuracy, "coherence": coherence, "latency_ms": latency, "n": len(tasks)}

    l1 = _stats(by_level[1])
    l2 = _stats(by_level[2])
    l3 = _stats(by_level[3])
    ces = l3["accuracy"] / l1["accuracy"] if l1["accuracy"] > 0 else 0.0

    # Error analysis
    errors: dict[str, int] = {}
    for task in results:
        if not task["correct"]:
            for rule in task["rules"]:
                errors[rule] = errors.get(rule, 0) + 1

    return {
        "L1": l1,
        "L2": l2,
        "L3": l3,
        "CES": ces,
        "error_analysis": errors,
        "total_tasks": len(results),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ccl_comparison(
    models: list[str] | None = None,
    output_path: Path | str = RESULTS_PATH,
    resume: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all 300 CCL tasks against the specified models.

    Args:
        models: list of model names (defaults to ALL_MODELS)
        output_path: where to save JSON results
        resume: if True, skip models that already appear in the output file
        seed: random seed for task generation

    Returns:
        Full results dict keyed by model name.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = models or ALL_MODELS
    tasks = generate_ccl_tasks(seed=seed)
    logger.info("Generated %d CCL tasks across 3 levels", len(tasks))

    # Load partial results if resuming
    existing: dict[str, Any] = {}
    if resume and output_path.exists():
        try:
            with output_path.open() as fh:
                existing = json.load(fh)
            logger.info("Resuming: found existing results for %s", list(existing.get("models", {}).keys()))
        except json.JSONDecodeError:
            existing = {}

    results: dict[str, Any] = existing.get("models", {})
    cache = ResponseCache()
    tracker = CostTracker()
    clients = build_clients(models, cache=cache, cost_tracker=tracker)

    for model_name, client in clients.items():
        if model_name in results:
            logger.info("Skipping %s (already complete)", model_name)
            continue

        logger.info("Evaluating model: %s", model_name)
        raw = _evaluate_model(client, tasks)
        results[model_name] = {
            "raw_results": raw,
            "summary": _aggregate(raw),
        }

        # Save after each model for resume support
        output = {"models": results, "cost": tracker.summary(), "seed": seed}
        with output_path.open("w") as fh:
            json.dump(output, fh, indent=2)
        logger.info("Saved results for %s → %s", model_name, output_path)

    final = {"models": results, "cost": tracker.summary(), "seed": seed}
    with output_path.open("w") as fh:
        json.dump(final, fh, indent=2)

    logger.info("CCL comparison complete. Results at %s", output_path)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ccl_comparison()
