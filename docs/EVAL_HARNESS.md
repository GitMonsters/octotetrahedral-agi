# AGI Evaluation Harness

A deterministic task generation, scoring, and regression tracking system for evaluating AGI components.

## Overview

The eval harness lives in the `eval_harness/` package and is designed around three principles:

1. **Reproducibility** — identical seeds always produce identical task sets and scores.
2. **Compositionality** — task families specifically stress generalisation beyond memorised patterns.
3. **Auditability** — all run artefacts are versioned, machine-readable, and stored persistently.

## Architecture

```
eval_harness/
├── generator.py   # Deterministic task generation (4 families, seeded RNG)
├── scorer.py      # Per-task + aggregate scoring with family breakdowns
├── tracker.py     # Run history persistence and regression comparison
├── cli.py         # CLI entrypoints: generate, evaluate, compare, trend
└── __main__.py    # Enables python -m eval_harness <command>
```

### Component responsibilities

| Module | Responsibility |
|--------|---------------|
| `generator` | Produce benchmark tasks from `random.Random(seed)`.  Serialize/deserialize JSONL with schema version. |
| `scorer`    | Score outputs against tasks.  Compute per-task, aggregate, and family-level metrics.  Handle partial credit and optional confidence fields. |
| `tracker`   | Save run artefacts (JSON).  Load run history.  Compare current vs baseline with configurable thresholds.  Render trend tables. |
| `cli`       | Expose `generate`, `evaluate`, `compare`, `trend` sub-commands with `--help` and sensible defaults. |

## Quickstart

### 1. Generate tasks

```bash
python -m eval_harness generate --seed 42 --num-tasks 80 --output tasks.jsonl
```

This produces `tasks.jsonl` — a JSONL file with 80 tasks spread across all four families.

### 2. Evaluate your system

Run your model on the tasks and produce an `outputs.jsonl` file with the format:

```jsonl
{"task_id": "compositional_0000", "answer": "yes"}
{"task_id": "sequence_0001", "answer": "48", "confidence": 0.92}
```

Then score it:

```bash
python -m eval_harness evaluate --tasks tasks.jsonl --outputs outputs.jsonl --tag mymodel-v1
```

You can also use the `--mock` flag to generate deterministic mock outputs for testing:

```bash
python -m eval_harness evaluate --tasks tasks.jsonl --mock --mock-score 0.7 --tag mock-run
```

### 3. Compare against baseline

```bash
# Compare the two most recent runs
python -m eval_harness compare

# Compare against a specific baseline (prefix of run ID)
python -m eval_harness compare --baseline abc123ef --threshold 0.02
```

Exit code `0` = no regression (or improvement); exit code `1` = regression detected.

### 4. View trend

```bash
python -m eval_harness trend --last 10
```

Sample output:

```
Timestamp              Tag                Seed  Overall   Correct  Tasks
------------------------------------------------------------------------
20260701T080000Z       baseline             42   0.7250        58/80
20260702T090000Z       v2-fix               42   0.7500        60/80
20260703T100000Z       v3-prompt-tune       42   0.8000        64/80
```

## Task Families

The harness ships with four task families that each stress a different aspect of compositional generalisation:

### `compositional`

Chain-rule deduction.  Given a set of premises like "A implies B; B implies C", asks whether a transitive conclusion follows.

- **Expected answer**: `"yes"` (chains are always valid by construction)
- **Metadata**: `depth`, `entities`, `predicate`
- **Scoring**: Exact string match (case-insensitive)

### `sequence`

Arithmetic and geometric sequence completion.  Given 4–6 terms, predict the next one.

- **Expected answer**: Integer as string, e.g. `"48"`
- **Metadata**: `kind` (`arithmetic` | `geometric`), `values`, `next`, `step_or_ratio`
- **Scoring**: Exact string match

### `analogy`

Concept analogies: *A is to B as C is to ?*

- **Expected answer**: Single word (lowercase)
- **Metadata**: `a`, `b`, `c`, `d`
- **Scoring**: Exact string match (case-insensitive)

### `pattern`

Symbolic grid transformation induction.  Given example input→output pairs, apply the same rule to a new row.

- **Expected answer**: Space-separated token string, e.g. `"G B R"`
- **Metadata**: `op` (`rotate` | `invert` | `shift` | `mirror`), `query`, `answer`, `examples_in`, `examples_out`
- **Scoring**: **Partial credit** — fraction of tokens correct (token-level alignment)

## Determinism Guarantees

- Task generation uses **only** `random.Random(seed)` — no global state, no OS entropy, no numpy.
- The same `seed` and `num_tasks` always yield byte-identical tasks on any CPython ≥ 3.8.
- Run artefacts include the seed and a `task_hash` (SHA-256 of all task IDs + content) so any change to the generator code or seed is immediately detectable.
- The `--mock` flag in `evaluate` also uses seeded randomness (`random.Random(task_id)`), making smoke-test runs fully reproducible.

## Run Artefact Schema (v1.0)

Each run is stored as a single JSON file in the runs directory:

```json
{
  "schema_version": "1.0",
  "run_id": "e18b185777c24c0bb50ecb1271f8daa9",
  "timestamp": "20260703T012812Z",
  "config": {
    "tasks_file": "tasks.jsonl",
    "outputs_file": "outputs.jsonl",
    "num_tasks": 80,
    "families": ["analogy", "compositional", "pattern", "sequence"]
  },
  "seed": 42,
  "task_hash": "6ca9bc273fef...",
  "overall": 0.8,
  "n_tasks": 80,
  "n_correct": 64,
  "family_scores": {
    "compositional": {"mean": 0.8, "n": 20, "n_correct": 16},
    "sequence":      {"mean": 0.9, "n": 20, "n_correct": 18},
    "analogy":       {"mean": 0.7, "n": 20, "n_correct": 14},
    "pattern":       {"mean": 0.7, "n": 20, "n_correct": 14}
  },
  "tag": "mymodel-v1",
  "extra": {}
}
```

## CLI Reference

```
python -m eval_harness generate
  --seed INT            Required. Integer seed for deterministic generation.
  --num-tasks INT       Total tasks (default: 80).
  --families STR        Comma-separated family names (default: all four).
  --output PATH         Output JSONL file (default: eval_harness_tasks.jsonl).

python -m eval_harness evaluate
  --tasks PATH          Input tasks JSONL (default: eval_harness_tasks.jsonl).
  --outputs PATH        System outputs JSONL (default: outputs.jsonl).
  --mock                Generate deterministic mock outputs (ignores --outputs).
  --mock-score FLOAT    Approximate correct fraction for mocks (default: 0.7).
  --seed INT            Override the seed stored in the tasks file.
  --runs-dir PATH       Directory for run artefacts (default: eval_runs/).
  --tag STR             Human-readable label for this run.

python -m eval_harness compare
  --baseline STR        Run ID prefix of baseline (default: second-most-recent).
  --current STR         Run ID prefix of current (default: most-recent).
  --threshold FLOAT     Delta threshold for reporting (default: 0.02).
  --runs-dir PATH       Directory of run artefacts (default: eval_runs/).

python -m eval_harness trend
  --last INT            Number of recent runs to show (default: 10).
  --runs-dir PATH       Directory of run artefacts (default: eval_runs/).
```

## Adding a New Task Family

1. Add a `_make_<family>(task_id, rng)` function in `eval_harness/generator.py` that returns a `TaskSpec`.
2. Register it in `_FAMILY_MAKERS` at the bottom of the same file.
3. Add the family name to the `FAMILIES` tuple.
4. Add a `_score_<family>(task, answer)` function in `eval_harness/scorer.py` that returns `(score: float, details: dict)`.
5. Register it in `_FAMILY_SCORERS`.
6. Add test coverage in `tests/test_eval_harness.py` under the relevant fixture classes.

## Adding a New Metric

The `aggregate_scores` function in `scorer.py` currently computes `mean`, `n`, and `n_correct` per family.  To add a new metric (e.g. confidence-weighted accuracy):

1. Update `_score_<family>` to return it in the `details` dict.
2. Update `aggregate_scores` to aggregate it in the `family_scores` dict.
3. Update the `RunScores` and `RunRecord` dataclasses if the metric should be persisted.

## Running Tests

```bash
python -m pytest tests/test_eval_harness.py -v
```

All 62 tests are deterministic and run in < 1 second with no external dependencies.
