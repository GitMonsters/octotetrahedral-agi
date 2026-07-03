"""AGI Evaluation Harness.

Provides deterministic task generation, scoring, and regression tracking
for evaluating AGI systems.

Modules:
    generator  — Task generation with seeded randomness and JSONL serialization
    scorer     — Scoring and metrics (per-task, aggregate, family-level)
    tracker    — Regression tracking with run history and baseline comparison
    cli        — CLI entrypoints: generate, evaluate, compare, trend

Quick start::

    python -m eval_harness generate --seed 42 --num-tasks 60 --output tasks.jsonl
    python -m eval_harness evaluate --tasks tasks.jsonl --outputs outputs.jsonl
    python -m eval_harness compare --baseline <run_id>
    python -m eval_harness trend
"""

from eval_harness.generator import generate_tasks, save_tasks, load_tasks
from eval_harness.scorer import score_tasks, aggregate_scores, TaskScore, RunScores
from eval_harness.tracker import save_run, load_runs, compare_runs, trend_summary, RunRecord

__all__ = [
    "generate_tasks",
    "save_tasks",
    "load_tasks",
    "score_tasks",
    "aggregate_scores",
    "TaskScore",
    "RunScores",
    "save_run",
    "load_runs",
    "compare_runs",
    "trend_summary",
    "RunRecord",
]
