"""Regression tracker for the AGI evaluation harness.

Persists run artefacts and compares the latest run against a chosen baseline,
reporting improvements/regressions with configurable thresholds.

Run artefact layout
-------------------
Each run is stored as a single JSON file in the *runs directory*::

    runs/
        <run_id>.json   # e.g. 20260703T012300Z_seed42.json

The file schema is versioned (``schema_version``) so old artefacts remain
readable after format changes.

Example::

    from eval_harness.tracker import save_run, load_runs, compare_runs, trend_summary

    record = RunRecord(...)
    save_run(record, runs_dir="runs/")

    runs = load_runs("runs/")
    cmp = compare_runs(current=runs[-1], baseline=runs[0])
    print(trend_summary(runs, last=5))
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RUN_SCHEMA_VERSION = "1.0"
_REGRESSION_THRESHOLD_DEFAULT = 0.02  # 2 percentage points


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Complete artefact for a single evaluation run.

    Attributes:
        schema_version: Artefact schema version; always ``"1.0"``.
        run_id:         Unique identifier (UUID4 hex).
        timestamp:      ISO-8601 UTC timestamp of the run.
        config:         Arbitrary config dict (seed, families, num_tasks, tag, …).
        seed:           The task-generation seed used for this run.
        task_hash:      SHA-256 of the task set (from :func:`~generator.task_set_hash`).
        overall:        Overall mean score (0–1).
        n_tasks:        Total tasks evaluated.
        n_correct:      Number of tasks with score == 1.0.
        family_scores:  Per-family ``{"mean": float, "n": int, "n_correct": int}``.
        tag:            Optional human-readable label, e.g. ``"baseline"`` or ``"v2"``.
    """

    schema_version: str
    run_id: str
    timestamp: str
    config: dict[str, Any]
    seed: int
    task_hash: str
    overall: float
    n_tasks: int
    n_correct: int
    family_scores: dict[str, dict[str, Any]]
    tag: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing a current run against a baseline.

    Attributes:
        delta_overall:      current.overall − baseline.overall
        improved:           ``True`` if delta_overall > threshold.
        regressed:          ``True`` if delta_overall < −threshold.
        threshold:          The configurable threshold used.
        family_deltas:      Per-family delta_mean values.
        family_regressions: Families where a significant regression occurred.
        family_improvements:Families where a significant improvement occurred.
        baseline_run_id:    ``run_id`` of the baseline.
        current_run_id:     ``run_id`` of the current run.
        summary:            Human-readable one-liner.
    """

    delta_overall: float
    improved: bool
    regressed: bool
    threshold: float
    family_deltas: dict[str, float]
    family_regressions: list[str]
    family_improvements: list[str]
    baseline_run_id: str
    current_run_id: str
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_filename(record: RunRecord) -> str:
    safe_tag = record.tag.replace(" ", "_") if record.tag else "run"
    return f"{record.timestamp}_{safe_tag}_{record.run_id[:8]}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_run_record(
    seed: int,
    task_hash: str,
    overall: float,
    n_tasks: int,
    n_correct: int,
    family_scores: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
    tag: str = "",
    run_id: str | None = None,
    timestamp: str | None = None,
) -> RunRecord:
    """Convenience constructor for :class:`RunRecord`.

    Fills in ``schema_version``, ``run_id``, and ``timestamp`` automatically
    when not provided.
    """
    return RunRecord(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id or uuid.uuid4().hex,
        timestamp=timestamp or _now_utc(),
        config=config or {},
        seed=seed,
        task_hash=task_hash,
        overall=overall,
        n_tasks=n_tasks,
        n_correct=n_correct,
        family_scores=family_scores,
        tag=tag,
    )


def save_run(record: RunRecord, runs_dir: str | Path = "runs/") -> Path:
    """Persist a :class:`RunRecord` to disk as a JSON file.

    Args:
        record:   The run record to save.
        runs_dir: Directory in which to write the file.

    Returns:
        Path to the written file.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    dest = runs_dir / _run_filename(record)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, indent=2)
    return dest


def load_runs(runs_dir: str | Path = "runs/") -> list[RunRecord]:
    """Load all run records from a directory, sorted by timestamp (oldest first).

    Args:
        runs_dir: Directory containing ``*.json`` run artefacts.

    Returns:
        List of :class:`RunRecord` objects sorted by ``timestamp``.
    """
    runs_dir = Path(runs_dir)
    records: list[RunRecord] = []
    if not runs_dir.exists():
        return records
    for p in sorted(runs_dir.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                data = json.load(fh)
            records.append(RunRecord(**data))
        except Exception:
            continue  # skip corrupt files
    records.sort(key=lambda r: r.timestamp)
    return records


def find_run(runs: list[RunRecord], run_id_prefix: str) -> RunRecord | None:
    """Find a run by a prefix of its ``run_id``."""
    for r in runs:
        if r.run_id.startswith(run_id_prefix):
            return r
    return None


def compare_runs(
    current: RunRecord,
    baseline: RunRecord,
    threshold: float = _REGRESSION_THRESHOLD_DEFAULT,
) -> ComparisonResult:
    """Compare a current run against a baseline.

    Args:
        current:   The latest run.
        baseline:  The reference run to compare against.
        threshold: Absolute delta threshold for reporting improvements/regressions.
                   Default: 0.02 (2 percentage points).

    Returns:
        :class:`ComparisonResult` with deltas and regression/improvement flags.
    """
    delta_overall = current.overall - baseline.overall
    improved = delta_overall > threshold
    regressed = delta_overall < -threshold

    # Per-family deltas
    family_deltas: dict[str, float] = {}
    family_regressions: list[str] = []
    family_improvements: list[str] = []

    all_families = set(current.family_scores) | set(baseline.family_scores)
    for fam in sorted(all_families):
        cur_mean = current.family_scores.get(fam, {}).get("mean", 0.0)
        base_mean = baseline.family_scores.get(fam, {}).get("mean", 0.0)
        delta = cur_mean - base_mean
        family_deltas[fam] = delta
        if delta < -threshold:
            family_regressions.append(fam)
        elif delta > threshold:
            family_improvements.append(fam)

    # Human-readable summary
    direction = "▲ improved" if improved else ("▼ regressed" if regressed else "— no change")
    summary = (
        f"{direction}: overall {baseline.overall:.4f} → {current.overall:.4f} "
        f"(Δ{delta_overall:+.4f})"
    )
    if family_regressions:
        summary += f"; regressions in: {', '.join(family_regressions)}"
    if family_improvements:
        summary += f"; improvements in: {', '.join(family_improvements)}"

    return ComparisonResult(
        delta_overall=delta_overall,
        improved=improved,
        regressed=regressed,
        threshold=threshold,
        family_deltas=family_deltas,
        family_regressions=family_regressions,
        family_improvements=family_improvements,
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        summary=summary,
    )


def trend_summary(
    runs: list[RunRecord],
    last: int = 10,
) -> str:
    """Return a human-readable trend table over the most recent *last* runs.

    Args:
        runs: All run records (sorted oldest→newest).
        last: How many recent runs to include in the table.

    Returns:
        A multi-line ASCII table string.
    """
    recent = runs[-last:] if len(runs) > last else runs
    if not recent:
        return "(no runs found)"

    lines = [
        f"{'Timestamp':<22} {'Tag':<16} {'Seed':>6} {'Overall':>8} {'Correct':>9} {'Tasks':>6}",
        "-" * 72,
    ]
    for r in recent:
        lines.append(
            f"{r.timestamp:<22} {r.tag or '—':<16} {r.seed:>6} "
            f"{r.overall:>8.4f} {r.n_correct:>9}/{r.n_tasks:<5}"
        )
    return "\n".join(lines)
