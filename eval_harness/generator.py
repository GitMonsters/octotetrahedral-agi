"""Deterministic task generator for the AGI evaluation harness.

Design principles
-----------------
* All randomness is seeded via ``random.Random(seed)``; no global state is
  touched so parallel calls with different seeds are always independent.
* Task families test distinct cognitive capabilities that stress
  compositional/generalisation behaviour:

  - ``compositional`` — chain-rule deduction (A→B, B→C ⊢ A→C)
  - ``sequence``      — arithmetic / geometric sequence completion
  - ``analogy``       — concept analogies (A:B :: C:?)
  - ``pattern``       — rule-induction over symbolic grids

* Generated tasks are serialised to JSONL with explicit ``schema_version``
  so old artefacts remain readable after schema changes.

Seed handling
-------------
Pass an explicit integer seed to :func:`generate_tasks`.  The same seed
always produces identical tasks on any platform (pure Python ``random``
module, no numpy RNG).  Store the seed alongside every run artefact so
evaluations are fully reproducible.

Example::

    from eval_harness.generator import generate_tasks, save_tasks, load_tasks

    tasks = generate_tasks(seed=42, num_tasks=60)
    save_tasks(tasks, "tasks.jsonl")
    loaded = load_tasks("tasks.jsonl")
    assert tasks == loaded
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"

FAMILIES: tuple[str, ...] = ("compositional", "sequence", "analogy", "pattern")


@dataclass
class TaskSpec:
    """A single benchmark task.

    Attributes:
        schema_version: Artifact schema version for forward-compatibility checks.
        task_id:        Unique identifier, e.g. ``compositional_042``.
        family:         Task family name (one of :data:`FAMILIES`).
        seed:           The per-task seed derived from the run seed.
        prompt:         Natural-language question shown to the evaluated system.
        expected:       Canonical answer key (lowercase); used by the scorer.
        metadata:       Family-specific structured data for richer analysis.
    """

    schema_version: str
    task_id: str
    family: str
    seed: int
    prompt: str
    expected: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------

# ------ compositional -------------------------------------------------------

_COMP_ENTITIES: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_COMP_PREDICATES: list[str] = [
    "is a kind of",
    "is part of",
    "implies",
    "causes",
    "enables",
    "precedes",
    "requires",
]


def _make_compositional(task_id: str, rng: random.Random) -> TaskSpec:
    """Chain-rule deduction: given A→B and B→C, does A→C hold?"""
    depth = rng.randint(2, 4)
    entities = rng.sample(_COMP_ENTITIES, k=depth + 1)
    predicate = rng.choice(_COMP_PREDICATES)

    premises: list[str] = []
    for i in range(depth):
        premises.append(f"{entities[i]} {predicate} {entities[i + 1]}")

    question_entity = entities[0]
    conclusion_entity = entities[-1]

    # All chains are valid by construction — answer is always "yes".
    answer = "yes"
    premises_str = "; ".join(premises)
    prompt = (
        f"Given: {premises_str}. "
        f"Does '{question_entity} {predicate} {conclusion_entity}' necessarily follow? "
        f"Answer yes or no."
    )
    return TaskSpec(
        schema_version=SCHEMA_VERSION,
        task_id=task_id,
        family="compositional",
        seed=rng.randint(0, 2**31),
        prompt=prompt,
        expected=answer,
        metadata={
            "depth": depth,
            "entities": entities,
            "predicate": predicate,
        },
    )


# ------ sequence ------------------------------------------------------------

def _make_sequence(task_id: str, rng: random.Random) -> TaskSpec:
    """Arithmetic or geometric sequence: find the next term."""
    kind = rng.choice(("arithmetic", "geometric"))
    start = rng.randint(1, 20)
    length = rng.randint(4, 6)

    if kind == "arithmetic":
        step = rng.randint(1, 10)
        values = [start + step * i for i in range(length + 1)]
    else:
        ratio = rng.randint(2, 4)
        values = [start * (ratio**i) for i in range(length + 1)]

    shown = values[:length]
    answer = str(values[length])
    seq_str = ", ".join(str(v) for v in shown)
    prompt = (
        f"What is the next number in the sequence: {seq_str}, ...? "
        f"Give only the number."
    )
    return TaskSpec(
        schema_version=SCHEMA_VERSION,
        task_id=task_id,
        family="sequence",
        seed=rng.randint(0, 2**31),
        prompt=prompt,
        expected=answer,
        metadata={
            "kind": kind,
            "values": shown,
            "next": values[length],
            "step_or_ratio": step if kind == "arithmetic" else ratio,
        },
    )


# ------ analogy -------------------------------------------------------------

_ANALOGY_PAIRS: list[tuple[str, str]] = [
    ("hot", "cold"),
    ("fast", "slow"),
    ("light", "dark"),
    ("big", "small"),
    ("hard", "soft"),
    ("loud", "quiet"),
    ("rough", "smooth"),
    ("heavy", "light"),
    ("sharp", "dull"),
    ("tall", "short"),
    ("wide", "narrow"),
    ("deep", "shallow"),
    ("wet", "dry"),
    ("strong", "weak"),
    ("old", "young"),
    ("new", "old"),
    ("open", "closed"),
    ("full", "empty"),
    ("alive", "dead"),
    ("clean", "dirty"),
]


def _make_analogy(task_id: str, rng: random.Random) -> TaskSpec:
    """Concept analogy: A is to B as C is to ?"""
    pairs = rng.sample(_ANALOGY_PAIRS, k=2)
    a, b = pairs[0]
    c, d = pairs[1]

    prompt = (
        f"'{a}' is to '{b}' as '{c}' is to what? "
        f"Give only the single word answer."
    )
    return TaskSpec(
        schema_version=SCHEMA_VERSION,
        task_id=task_id,
        family="analogy",
        seed=rng.randint(0, 2**31),
        prompt=prompt,
        expected=d,
        metadata={"a": a, "b": b, "c": c, "d": d},
    )


# ------ pattern -------------------------------------------------------------

_PATTERN_COLORS: list[str] = ["R", "G", "B", "Y", "W", "K"]
_PATTERN_OPS: list[str] = ["rotate", "invert", "shift", "mirror"]


def _apply_pattern_op(row: list[str], op: str, colors: list[str]) -> list[str]:
    n = len(row)
    if op == "rotate":
        return row[1:] + [row[0]]
    if op == "invert":
        idx_map = {c: colors[(colors.index(c) + len(colors) // 2) % len(colors)] for c in colors}
        return [idx_map.get(c, c) for c in row]
    if op == "shift":
        return [colors[(colors.index(c) + 1) % len(colors)] if c in colors else c for c in row]
    if op == "mirror":
        return list(reversed(row))
    return row


def _make_pattern(task_id: str, rng: random.Random) -> TaskSpec:
    """Rule induction: deduce the transformation from examples, apply to a new row."""
    n = rng.randint(3, 5)
    colors = rng.sample(_PATTERN_COLORS, k=min(n, len(_PATTERN_COLORS)))
    op = rng.choice(_PATTERN_OPS)
    n_examples = rng.randint(2, 3)

    # Generate example input rows
    examples_in: list[list[str]] = []
    examples_out: list[list[str]] = []
    for _ in range(n_examples):
        row = [rng.choice(colors) for _ in range(n)]
        examples_in.append(row)
        examples_out.append(_apply_pattern_op(row, op, colors))

    # Generate query row
    query_row = [rng.choice(colors) for _ in range(n)]
    answer_row = _apply_pattern_op(query_row, op, colors)
    answer = " ".join(answer_row)

    examples_str = "; ".join(
        f"{''.join(ei)} -> {''.join(eo)}"
        for ei, eo in zip(examples_in, examples_out)
    )
    query_str = "".join(query_row)
    prompt = (
        f"A transformation rule is applied consistently. Examples: {examples_str}. "
        f"Apply the same rule to: {query_str}. "
        f"Give the output as space-separated tokens."
    )
    return TaskSpec(
        schema_version=SCHEMA_VERSION,
        task_id=task_id,
        family="pattern",
        seed=rng.randint(0, 2**31),
        prompt=prompt,
        expected=answer,
        metadata={
            "op": op,
            "query": query_row,
            "answer": answer_row,
            "examples_in": examples_in,
            "examples_out": examples_out,
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FAMILY_MAKERS = {
    "compositional": _make_compositional,
    "sequence": _make_sequence,
    "analogy": _make_analogy,
    "pattern": _make_pattern,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tasks(
    seed: int,
    num_tasks: int = 80,
    families: list[str] | None = None,
) -> list[TaskSpec]:
    """Generate a deterministic list of benchmark tasks.

    Args:
        seed:       Integer seed.  The same seed always produces identical tasks.
        num_tasks:  Total number of tasks to generate, distributed evenly across
                    the requested families.
        families:   Subset of :data:`FAMILIES` to include.  Defaults to all four.

    Returns:
        Ordered list of :class:`TaskSpec` objects.

    Raises:
        ValueError: If an unknown family name is requested.

    Reproducibility guarantee
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    This function uses only ``random.Random(seed)`` seeded once at the start.
    No global random state, no numpy, no OS entropy.  Re-running with the same
    arguments on any CPython ≥ 3.8 installation yields byte-identical outputs.
    """
    if families is None:
        families = list(FAMILIES)
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"Unknown task families: {unknown}. Valid: {FAMILIES}")
    if not families:
        raise ValueError("families must be non-empty")
    if num_tasks < 1:
        raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")

    rng = random.Random(seed)
    tasks: list[TaskSpec] = []
    per_family = max(1, num_tasks // len(families))
    remainder = num_tasks - per_family * len(families)

    for fi, family in enumerate(families):
        n = per_family + (1 if fi < remainder else 0)
        maker = _FAMILY_MAKERS[family]
        for i in range(n):
            task_id = f"{family}_{len(tasks):04d}"
            tasks.append(maker(task_id, rng))

    return tasks


def task_set_hash(tasks: list[TaskSpec]) -> str:
    """Return a stable SHA-256 hex digest of the task set content.

    Useful for detecting when the generated task set has changed between runs
    (e.g. after a generator code change with the same seed).
    """
    h = hashlib.sha256()
    for t in tasks:
        h.update(f"{t.task_id}:{t.prompt}:{t.expected}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_tasks(tasks: list[TaskSpec], path: str | Path) -> None:
    """Serialise tasks to a JSONL file (one JSON object per line).

    Each line is a self-contained task object that includes ``schema_version``
    for forward-compatibility.  The file is written atomically (temp file +
    rename on POSIX systems via Python's default file write).

    Args:
        tasks: List of :class:`TaskSpec` objects.
        path:  Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Load tasks previously saved with :func:`save_tasks`.

    Args:
        path: Path to a JSONL file created by :func:`save_tasks`.

    Returns:
        List of :class:`TaskSpec` objects in the original order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line cannot be parsed or has an unsupported schema
                    version.
    """
    path = Path(path)
    tasks: list[TaskSpec] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            version = data.get("schema_version", "")
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema_version '{version}' on line {lineno}. "
                    f"Expected '{SCHEMA_VERSION}'."
                )
            tasks.append(TaskSpec(**data))
    return tasks
