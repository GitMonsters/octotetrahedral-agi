"""CLI for the AGI evaluation harness.

Commands
--------
generate
    Generate a deterministic task set and write it to disk.

evaluate
    Score pre-computed system outputs against generated tasks and save the
    run artefact.

compare
    Compare the latest run (or a specified run) against a baseline run and
    report improvements / regressions.

trend
    Display a trend table over the most recent N runs.

Usage examples::

    python -m eval_harness generate --seed 42 --num-tasks 80 --output tasks.jsonl
    python -m eval_harness evaluate --tasks tasks.jsonl --outputs out.jsonl --tag baseline
    python -m eval_harness compare --baseline <run_id_prefix>
    python -m eval_harness trend --last 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_TASKS_PATH = Path("eval_harness_tasks.jsonl")
DEFAULT_RUNS_DIR = Path("eval_runs/")

# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate tasks and write to a JSONL file."""
    from eval_harness.generator import FAMILIES, generate_tasks, save_tasks, task_set_hash

    families: list[str] | None = None
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]

    try:
        tasks = generate_tasks(seed=args.seed, num_tasks=args.num_tasks, families=families)
    except ValueError as exc:
        logger.error("Task generation failed: %s", exc)
        return 1

    output = Path(args.output)
    save_tasks(tasks, output)
    thash = task_set_hash(tasks)

    families_used = sorted({t.family for t in tasks})
    print(f"Generated {len(tasks)} tasks → {output}")
    print(f"  Seed:       {args.seed}")
    print(f"  Families:   {', '.join(families_used)}")
    print(f"  Task hash:  {thash}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score outputs and persist a run record."""
    from eval_harness.generator import load_tasks, task_set_hash
    from eval_harness.scorer import aggregate_scores, score_tasks
    from eval_harness.tracker import make_run_record, save_run

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        logger.error("Tasks file not found: %s", tasks_path)
        return 1

    try:
        tasks = load_tasks(tasks_path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Failed to load tasks: %s", exc)
        return 1

    # Load outputs (JSONL or JSON array)
    outputs_path = Path(args.outputs)
    if not outputs_path.exists():
        if args.mock:
            outputs = _generate_mock_outputs(tasks, score=args.mock_score)
        else:
            logger.error(
                "Outputs file not found: %s. Use --mock to generate mock outputs.", outputs_path
            )
            return 1
    else:
        outputs = _load_outputs(outputs_path)

    task_scores = score_tasks(tasks, outputs)
    run_scores = aggregate_scores(task_scores)
    thash = task_set_hash(tasks)

    # Reconstruct seed from tasks (all tasks store the run seed in config)
    seed = tasks[0].seed if tasks else 0
    # Try to read seed from the tasks file header line if needed;
    # fall back to using args.seed if provided
    if hasattr(args, "seed") and args.seed is not None:
        seed = args.seed

    config: dict[str, Any] = {
        "tasks_file": str(tasks_path),
        "outputs_file": str(outputs_path) if not args.mock else "mock",
        "num_tasks": len(tasks),
        "families": sorted({t.family for t in tasks}),
    }

    record = make_run_record(
        seed=seed,
        task_hash=thash,
        overall=run_scores.overall,
        n_tasks=run_scores.n_tasks,
        n_correct=run_scores.n_correct,
        family_scores=run_scores.family_scores,
        config=config,
        tag=args.tag or "",
    )

    runs_dir = Path(args.runs_dir)
    saved_path = save_run(record, runs_dir)

    print(f"Evaluation complete — run ID: {record.run_id}")
    print(f"  Overall score: {run_scores.overall:.4f}  ({run_scores.n_correct}/{run_scores.n_tasks} correct)")
    for fam, fs in sorted(run_scores.family_scores.items()):
        print(f"  {fam:<20} {fs['mean']:.4f}  ({fs['n_correct']}/{fs['n']} correct)")
    print(f"  Artefact: {saved_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare the current run to a baseline and report regressions."""
    from eval_harness.tracker import compare_runs, find_run, load_runs

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir)
    if not runs:
        logger.error("No run records found in %s", runs_dir)
        return 1

    # Resolve baseline
    if args.baseline:
        baseline = find_run(runs, args.baseline)
        if baseline is None:
            logger.error("Baseline run '%s' not found in %s", args.baseline, runs_dir)
            return 1
    else:
        if len(runs) < 2:
            logger.error("Need at least 2 runs to compare.  Specify --baseline explicitly.")
            return 1
        baseline = runs[-2]

    # Resolve current
    if args.current:
        current = find_run(runs, args.current)
        if current is None:
            logger.error("Current run '%s' not found in %s", args.current, runs_dir)
            return 1
    else:
        current = runs[-1]

    result = compare_runs(current, baseline, threshold=args.threshold)

    print(result.summary)
    print(f"\n  Baseline : {baseline.run_id[:8]}  {baseline.timestamp}  tag={baseline.tag or '—'}")
    print(f"  Current  : {current.run_id[:8]}  {current.timestamp}  tag={current.tag or '—'}")
    print(f"\n  Family deltas (threshold ±{args.threshold:.3f}):")
    for fam, delta in sorted(result.family_deltas.items()):
        flag = " ▼" if fam in result.family_regressions else (" ▲" if fam in result.family_improvements else "")
        print(f"    {fam:<20} {delta:+.4f}{flag}")

    return 1 if result.regressed else 0


def cmd_trend(args: argparse.Namespace) -> int:
    """Print a trend table of recent runs."""
    from eval_harness.tracker import load_runs, trend_summary

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir)
    print(trend_summary(runs, last=args.last))
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_outputs(path: Path) -> list[dict[str, Any]]:
    """Load outputs from a JSONL or JSON-array file."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    # Try JSON array first
    if raw.startswith("["):
        return json.loads(raw)
    # Fall back to JSONL
    outputs = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            outputs.append(json.loads(line))
    return outputs


def _generate_mock_outputs(
    tasks: Any,
    score: float = 0.7,
) -> list[dict[str, Any]]:
    """Generate deterministic mock outputs for smoke-testing.

    Uses ``random.Random`` seeded on task_id so results are reproducible.
    The probability of a correct answer is approximately *score*.
    """
    import random

    outputs: list[dict[str, Any]] = []
    for task in tasks:
        rng = random.Random(task.task_id)
        if rng.random() < score:
            answer = task.expected  # correct
        else:
            answer = "__wrong__"
        outputs.append({"task_id": task.task_id, "answer": answer})
    return outputs


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="eval_harness",
        description="AGI Evaluation Harness — deterministic task generation, scoring, regression tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m eval_harness generate --seed 42 --num-tasks 80\n"
            "  python -m eval_harness evaluate --tasks tasks.jsonl --mock\n"
            "  python -m eval_harness compare --baseline abc123\n"
            "  python -m eval_harness trend --last 5\n"
        ),
    )
    root.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )

    sub = root.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- generate ----------------------------------------------------------
    gen_p = sub.add_parser(
        "generate",
        help="Generate a deterministic task set and write to JSONL.",
        description="Generate benchmark tasks from a seeded random generator.",
    )
    gen_p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Integer seed for deterministic task generation.",
    )
    gen_p.add_argument(
        "--num-tasks",
        type=int,
        default=80,
        help="Total number of tasks to generate, distributed across families (default: 80).",
    )
    gen_p.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-separated list of task families to include. "
            "Choices: compositional, sequence, analogy, pattern. "
            "Default: all four."
        ),
    )
    gen_p.add_argument(
        "--output",
        default=str(DEFAULT_TASKS_PATH),
        help=f"Output JSONL file (default: {DEFAULT_TASKS_PATH}).",
    )
    gen_p.set_defaults(func=cmd_generate)

    # ---- evaluate ----------------------------------------------------------
    eval_p = sub.add_parser(
        "evaluate",
        help="Score system outputs against tasks and save a run artefact.",
        description="Score outputs from an evaluated system against the task set.",
    )
    eval_p.add_argument(
        "--tasks",
        default=str(DEFAULT_TASKS_PATH),
        help=f"JSONL file of tasks (default: {DEFAULT_TASKS_PATH}).",
    )
    eval_p.add_argument(
        "--outputs",
        default="outputs.jsonl",
        help="JSONL file of system outputs with 'task_id' and 'answer' fields (default: outputs.jsonl).",
    )
    eval_p.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock outputs (for testing; ignores --outputs file).",
    )
    eval_p.add_argument(
        "--mock-score",
        type=float,
        default=0.7,
        help="Approximate fraction of correct answers in mock outputs (default: 0.7).",
    )
    eval_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Task-generation seed (overrides the value stored in the tasks file).",
    )
    eval_p.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help=f"Directory for run artefacts (default: {DEFAULT_RUNS_DIR}).",
    )
    eval_p.add_argument(
        "--tag",
        default="",
        help="Human-readable tag for this run (e.g. 'baseline', 'v2-fix').",
    )
    eval_p.set_defaults(func=cmd_evaluate)

    # ---- compare -----------------------------------------------------------
    cmp_p = sub.add_parser(
        "compare",
        help="Compare the current run against a baseline; exit 1 on regression.",
        description=(
            "Compare two runs and report improvements/regressions. "
            "Returns exit code 1 if a regression is detected."
        ),
    )
    cmp_p.add_argument(
        "--baseline",
        default=None,
        help=(
            "Prefix of the baseline run ID (from the artefact filename). "
            "If omitted, the second-most-recent run is used."
        ),
    )
    cmp_p.add_argument(
        "--current",
        default=None,
        help=(
            "Prefix of the current run ID. "
            "If omitted, the most-recent run is used."
        ),
    )
    cmp_p.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Absolute delta threshold for regression/improvement (default: 0.02).",
    )
    cmp_p.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help=f"Directory of run artefacts (default: {DEFAULT_RUNS_DIR}).",
    )
    cmp_p.set_defaults(func=cmd_compare)

    # ---- trend -------------------------------------------------------------
    trend_p = sub.add_parser(
        "trend",
        help="Print a trend table over recent runs.",
        description="Display overall score trends across the most recent runs.",
    )
    trend_p.add_argument(
        "--last",
        type=int,
        default=10,
        help="Number of recent runs to include (default: 10).",
    )
    trend_p.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help=f"Directory of run artefacts (default: {DEFAULT_RUNS_DIR}).",
    )
    trend_p.set_defaults(func=cmd_trend)

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
