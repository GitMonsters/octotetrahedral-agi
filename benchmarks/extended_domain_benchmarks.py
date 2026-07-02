"""Extended domain benchmarks: test all models across diverse domains.

Domains:
  - reasoning  — logic puzzles, math problems, proof verification
  - language   — translation, summarisation, generation quality
  - spatial    — geometry, grid manipulation, route planning
  - planning   — multi-step decomposition, resource optimisation
  - multi      — tasks combining two or more domains

Metrics per domain: accuracy, quality_score, latency_ms, coherence (where available)
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

RESULTS_PATH = Path("benchmarks/results/extended_domain_results.json")

# ---------------------------------------------------------------------------
# Task bank
# ---------------------------------------------------------------------------

_DOMAIN_TASKS: dict[str, list[dict[str, Any]]] = {
    "reasoning": [
        {"prompt": "If all A are B, and all B are C, are all A also C?", "expected_key": "yes"},
        {"prompt": "What is 17 × 13?", "expected_key": "221"},
        {"prompt": "Is the statement 'P AND (NOT P)' always false?", "expected_key": "yes"},
        {"prompt": "Solve: x + 5 = 12.", "expected_key": "7"},
        {"prompt": "Are two sets with the same elements equal?", "expected_key": "yes"},
        {"prompt": "What is the next prime after 11?", "expected_key": "13"},
        {"prompt": "If today is Monday, what day is it in 8 days?", "expected_key": "tuesday"},
        {"prompt": "A train travels 60 km/h for 2.5 hours. How far?", "expected_key": "150"},
        {"prompt": "Is 1024 a power of 2?", "expected_key": "yes"},
        {"prompt": "What is 15% of 200?", "expected_key": "30"},
    ],
    "language": [
        {"prompt": "Translate to French: 'The sky is blue.'", "expected_key": "ciel"},
        {"prompt": "Translate to Spanish: 'Good morning.'", "expected_key": "buenos"},
        {"prompt": "Summarise in one sentence: 'Machine learning uses data to train models that make predictions without explicit programming rules.'", "expected_key": "train"},
        {"prompt": "Generate a synonym for 'happy'.", "expected_key": "joyful"},
        {"prompt": "Translate to French: 'I love coding.'", "expected_key": "code"},
        {"prompt": "Is 'The cat sat on the mat' grammatically correct?", "expected_key": "yes"},
        {"prompt": "Translate to Spanish: 'Where is the library?'", "expected_key": "biblioteca"},
        {"prompt": "Summarise: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make glucose and oxygen.'", "expected_key": "plant"},
        {"prompt": "What is the plural of 'mouse'?", "expected_key": "mice"},
        {"prompt": "Generate an antonym for 'fast'.", "expected_key": "slow"},
    ],
    "spatial": [
        {"prompt": "How many degrees in a right angle?", "expected_key": "90"},
        {"prompt": "What is the area of a 4×5 rectangle?", "expected_key": "20"},
        {"prompt": "If you face North and turn 180°, which direction do you face?", "expected_key": "south"},
        {"prompt": "How many sides does a hexagon have?", "expected_key": "6"},
        {"prompt": "What is the perimeter of a square with side 7?", "expected_key": "28"},
        {"prompt": "In a 3×3 grid, how many cells are there?", "expected_key": "9"},
        {"prompt": "What is the diagonal length of a 3-4-5 right triangle?", "expected_key": "5"},
        {"prompt": "If you move 3 steps East then 3 steps West, where are you?", "expected_key": "start"},
        {"prompt": "How many edges does a cube have?", "expected_key": "12"},
        {"prompt": "What is the volume of a 2×2×2 cube?", "expected_key": "8"},
    ],
    "planning": [
        {"prompt": "List the steps to make a sandwich in order.", "expected_key": "bread"},
        {"prompt": "You have tasks A (depends on nothing), B (depends on A), C (depends on A). What order?", "expected_key": "a"},
        {"prompt": "To bake a cake you need flour, eggs, and sugar. You have flour and eggs. What are you missing?", "expected_key": "sugar"},
        {"prompt": "A project has 3 phases: design, build, test. In what order must they run?", "expected_key": "design"},
        {"prompt": "If you have 10 minutes and need to do a 6-min task and a 5-min task, which must you skip?", "expected_key": "5"},
        {"prompt": "Decompose 'deploy a web app' into the first 3 sub-tasks.", "expected_key": "build"},
        {"prompt": "You need to buy groceries before cooking. Which comes first?", "expected_key": "groceries"},
        {"prompt": "To optimise throughput, should you run independent tasks in parallel or serial?", "expected_key": "parallel"},
        {"prompt": "What is the critical path in a project with tasks A→B→D and A→C→D, where B takes 3 days and C takes 5 days?", "expected_key": "c"},
        {"prompt": "If step 3 depends on step 2 which depends on step 1, can you skip step 2?", "expected_key": "no"},
    ],
    "multi": [
        {"prompt": "Translate 'rotate 90 degrees' to French AND describe the resulting spatial transformation.", "expected_key": "pivoter"},
        {"prompt": "Plan a 3-step logic proof for: all X are Y, all Y are Z, therefore all X are Z.", "expected_key": "z"},
        {"prompt": "Summarise AND solve: 'If 2x + 4 = 10, what is x?'", "expected_key": "3"},
        {"prompt": "Describe the spatial layout of a chessboard in English AND Spanish.", "expected_key": "board"},
        {"prompt": "Plan the steps to build a reasoning system that handles AND-composed rules.", "expected_key": "compos"},
        {"prompt": "Translate to French AND calculate: 'What is 6 times 7?'", "expected_key": "42"},
        {"prompt": "Spatial AND planning: draw a 3-step grid path from (0,0) to (2,2) avoiding (1,1).", "expected_key": "path"},
        {"prompt": "Reasoning AND language: prove that 'No A is B' contradicts 'Some A is B', in Spanish.", "expected_key": "ning"},
        {"prompt": "Plan AND spatial: optimise the order of visiting 3 cities in a triangle.", "expected_key": "triangle"},
        {"prompt": "Language AND reasoning: write a syllogism in French.", "expected_key": "donc"},
    ],
}


def _generate_domain_tasks() -> list[dict[str, Any]]:
    tasks = []
    for domain, bank in _DOMAIN_TASKS.items():
        for i, t in enumerate(bank):
            tasks.append({
                "task_id": f"domain_{domain}_{i:03d}",
                "domain": domain,
                "prompt": t["prompt"],
                "expected_key": t["expected_key"],
            })
    return tasks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_response(response: dict[str, Any], expected_key: str) -> dict[str, float]:
    """Compute accuracy and quality_score from a model response."""
    correct = response.get("correct")
    answer = str(response.get("answer", "")).lower()

    if correct is None:
        correct = expected_key.lower() in answer

    quality = 1.0 if correct else 0.0
    return {"accuracy": float(correct), "quality_score": quality}


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model_domains(
    client: ModelClient,
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for task in tasks:
        t0 = time.perf_counter()
        response = client.call(task["prompt"], task_signal=task["domain"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        scores = _score_response(response, task["expected_key"])
        results.append({
            "task_id": task["task_id"],
            "domain": task["domain"],
            "correct": bool(scores["accuracy"]),
            "accuracy": scores["accuracy"],
            "quality_score": scores["quality_score"],
            "coherence": response.get("coherence"),
            "latency_ms": response.get("latency_ms", elapsed_ms),
            "model": client.model,
        })
    return results


def _aggregate_domains(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)

    summary: dict[str, Any] = {}
    for domain, items in by_domain.items():
        accuracy = sum(i["accuracy"] for i in items) / len(items)
        quality = sum(i["quality_score"] for i in items) / len(items)
        latency = sum(i["latency_ms"] for i in items) / len(items)
        coherence_vals = [i["coherence"] for i in items if i["coherence"] is not None]
        coherence = sum(coherence_vals) / len(coherence_vals) if coherence_vals else None
        summary[domain] = {
            "accuracy": accuracy,
            "quality_score": quality,
            "latency_ms": latency,
            "coherence": coherence,
            "n": len(items),
        }
    return summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_extended_benchmarks(
    models: list[str] | None = None,
    output_path: Path | str = RESULTS_PATH,
    resume: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = models or ALL_MODELS
    tasks = _generate_domain_tasks()
    logger.info("Running extended domain benchmarks: %d tasks × %d models", len(tasks), len(models))

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
        logger.info("Evaluating %s across domains …", model_name)
        raw = _evaluate_model_domains(client, tasks)
        results[model_name] = {
            "raw_results": raw,
            "domain_summary": _aggregate_domains(raw),
        }
        output = {"models": results, "cost": tracker.summary()}
        with output_path.open("w") as fh:
            json.dump(output, fh, indent=2)

    final = {"models": results, "cost": tracker.summary()}
    with output_path.open("w") as fh:
        json.dump(final, fh, indent=2)

    logger.info("Extended domain benchmarks complete → %s", output_path)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_extended_benchmarks()
