"""Scoring module for the AGI evaluation harness.

Evaluates system outputs against expected task answers and computes
per-task scores, aggregate scores, and family-level breakdowns.

Scoring strategy
----------------
* ``compositional`` — exact string match (expected is always ``"yes"`` or ``"no"``)
* ``sequence``      — exact numeric match (strip whitespace, compare as strings)
* ``analogy``       — exact word match (case-insensitive, stripped)
* ``pattern``       — token-level partial credit: fraction of tokens correct

Confidence / uncertainty
------------------------
Outputs may optionally include a ``confidence`` field (float 0–1).  When
present it is stored in the :class:`TaskScore` but does not alter the
primary score, preserving comparability across systems that do and do not
report confidence.

Example::

    from eval_harness.scorer import score_tasks, aggregate_scores

    outputs = [{"task_id": "sequence_0000", "answer": "24"}]
    task_scores = score_tasks(tasks, outputs)
    run_scores = aggregate_scores(task_scores)
    print(run_scores.overall)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eval_harness.generator import TaskSpec

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    """Score for a single task.

    Attributes:
        task_id:    Matches the corresponding :class:`~generator.TaskSpec`.
        family:     Task family (for grouping in breakdowns).
        score:      Primary score in [0.0, 1.0].  Full credit = 1.0.
        correct:    ``True`` if score == 1.0.
        answer:     The answer provided by the evaluated system.
        expected:   The expected canonical answer.
        confidence: Optional confidence reported by the system (0–1).
        details:    Free-form dict for family-specific extra fields.
    """

    task_id: str
    family: str
    score: float
    correct: bool
    answer: str
    expected: str
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunScores:
    """Aggregate scores for a complete evaluation run.

    Attributes:
        overall:        Mean score across all tasks.
        n_tasks:        Total number of tasks scored.
        n_correct:      Number of tasks with score == 1.0.
        family_scores:  Per-family breakdown ``{family: {"mean": float, "n": int}}``.
        task_scores:    Individual :class:`TaskScore` objects.
    """

    overall: float
    n_tasks: int
    n_correct: int
    family_scores: dict[str, dict[str, Any]]
    task_scores: list[TaskScore]


# ---------------------------------------------------------------------------
# Per-family scoring functions
# ---------------------------------------------------------------------------


def _score_compositional(task: TaskSpec, answer: str) -> tuple[float, dict[str, Any]]:
    """Exact match on 'yes'/'no' answer."""
    normalised = answer.strip().lower().rstrip(".")
    correct = normalised == task.expected.lower()
    return (1.0 if correct else 0.0), {"normalised_answer": normalised}


def _score_sequence(task: TaskSpec, answer: str) -> tuple[float, dict[str, Any]]:
    """Exact match after stripping whitespace and punctuation."""
    normalised = answer.strip().rstrip(".").strip()
    correct = normalised == task.expected
    return (1.0 if correct else 0.0), {"normalised_answer": normalised}


def _score_analogy(task: TaskSpec, answer: str) -> tuple[float, dict[str, Any]]:
    """Case-insensitive exact match for the single-word answer."""
    normalised = answer.strip().lower().rstrip(".")
    correct = normalised == task.expected.lower()
    return (1.0 if correct else 0.0), {"normalised_answer": normalised}


def _score_pattern(task: TaskSpec, answer: str) -> tuple[float, dict[str, Any]]:
    """Token-level partial credit: fraction of tokens that are correct."""
    expected_tokens = task.expected.split()
    answer_tokens = answer.strip().split()

    if not expected_tokens:
        return 1.0, {}

    # Align by position; pad shorter with empty strings
    n = max(len(expected_tokens), len(answer_tokens))
    matched = sum(
        1
        for i in range(n)
        if i < len(expected_tokens)
        and i < len(answer_tokens)
        and answer_tokens[i].upper() == expected_tokens[i].upper()
    )
    score = matched / len(expected_tokens)
    return score, {
        "expected_tokens": expected_tokens,
        "answer_tokens": answer_tokens,
        "matched": matched,
    }


_FAMILY_SCORERS = {
    "compositional": _score_compositional,
    "sequence": _score_sequence,
    "analogy": _score_analogy,
    "pattern": _score_pattern,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_task(task: TaskSpec, output: dict[str, Any]) -> TaskScore:
    """Score a single task against a system output.

    Args:
        task:   The :class:`~generator.TaskSpec` to score.
        output: A dict with at minimum ``"task_id"`` and ``"answer"`` keys.
                An optional ``"confidence"`` key (float 0–1) is carried through
                without affecting the score.

    Returns:
        A :class:`TaskScore` with ``score`` in [0.0, 1.0].
    """
    answer = str(output.get("answer", ""))
    confidence = output.get("confidence")

    scorer = _FAMILY_SCORERS.get(task.family)
    if scorer is None:
        # Unknown family: fall back to exact match
        score = 1.0 if answer.strip().lower() == task.expected.lower() else 0.0
        details: dict[str, Any] = {"warning": f"unknown family '{task.family}', used exact match"}
    else:
        score, details = scorer(task, answer)

    return TaskScore(
        task_id=task.task_id,
        family=task.family,
        score=score,
        correct=score >= 1.0,
        answer=answer,
        expected=task.expected,
        confidence=float(confidence) if confidence is not None else None,
        details=details,
    )


def score_tasks(
    tasks: list[TaskSpec],
    outputs: list[dict[str, Any]],
) -> list[TaskScore]:
    """Score a batch of tasks against the corresponding outputs.

    ``tasks`` and ``outputs`` are matched by ``task_id``.  Tasks with no
    corresponding output receive a score of 0.0 with ``answer=""``.

    Args:
        tasks:   Tasks to score.
        outputs: One dict per task, each with ``"task_id"`` and ``"answer"``.

    Returns:
        List of :class:`TaskScore` in the same order as *tasks*.
    """
    output_map = {str(o.get("task_id", "")): o for o in outputs}
    results: list[TaskScore] = []
    for task in tasks:
        out = output_map.get(task.task_id, {"task_id": task.task_id, "answer": ""})
        results.append(score_task(task, out))
    return results


def aggregate_scores(task_scores: list[TaskScore]) -> RunScores:
    """Compute aggregate and family-level scores from per-task scores.

    Args:
        task_scores: List of :class:`TaskScore` objects.

    Returns:
        :class:`RunScores` with ``overall``, ``n_tasks``, ``n_correct``,
        and per-family breakdowns.
    """
    if not task_scores:
        return RunScores(
            overall=0.0,
            n_tasks=0,
            n_correct=0,
            family_scores={},
            task_scores=[],
        )

    overall = sum(ts.score for ts in task_scores) / len(task_scores)
    n_correct = sum(1 for ts in task_scores if ts.correct)

    by_family: dict[str, list[TaskScore]] = {}
    for ts in task_scores:
        by_family.setdefault(ts.family, []).append(ts)

    family_scores: dict[str, dict[str, Any]] = {}
    for family, scores in by_family.items():
        mean = sum(ts.score for ts in scores) / len(scores)
        family_scores[family] = {
            "mean": mean,
            "n": len(scores),
            "n_correct": sum(1 for ts in scores if ts.correct),
        }

    return RunScores(
        overall=overall,
        n_tasks=len(task_scores),
        n_correct=n_correct,
        family_scores=family_scores,
        task_scores=task_scores,
    )
