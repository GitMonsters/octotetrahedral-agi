# 🧩 RE-ARC Bench — 125/125 (100%)

## A Higher Level of Abstraction

**RE-ARC** (Reverse-Engineering the Abstraction and Reasoning Corpus) is fundamentally harder than ARC-AGI-1 or ARC-AGI-2. While those benchmarks test whether a solver can deduce transformation rules from hand-crafted examples, RE-ARC tests whether the solver truly understands the underlying *abstraction*.

### Why RE-ARC Should Be Harder

| Property | ARC-AGI-1/2 | RE-ARC |
|----------|-------------|--------|
| Task source | Hand-crafted | Procedurally generated |
| Test inputs | Fixed | Infinite variations possible |
| Pattern matching | Can work | Cannot work |
| Memorization | Possible (overfit) | Impossible |
| True generalization | Not required | **Required** |

Each RE-ARC task has an underlying *generator program* — the solver must capture the essence of that program, not just recognize surface patterns. Color permutations and rotation/flip transforms ensure verifiers can't trivially solve them.

### RE-ARC Bench Curation

This benchmark was curated by David Lu:
1. Removed all tasks solvable by the icecuber solver (same curation as ARC-AGI-2)
2. Selected the most complex tasks by verifier line count
3. Applied color permutations and transforms

**Result: Difficulty roughly tracks ARC-AGI-2, but with infinite fresh test instances.**

---

## Score: 125/125 (100%)

| Metric | Value |
|--------|-------|
| Tasks Solved | 125/125 |
| Success Rate | 100% |
| Total Predictions | 237 |
| Avg Predictions/Task | 1.90 |
| Avg Solver Lines | 111 |

## Methodology

Same as ARC-AGI-1/2: **LLM-guided program synthesis** using Claude Opus 4.6.

Each solver is a standalone Python `transform(grid)` function that captures the abstract rule — not a pattern-matched heuristic. This is why TranscendPlexity succeeds on RE-ARC: the program synthesis approach inherently discovers the underlying algorithm, which generalizes to any procedurally-generated instance.

## Files

- `submission.json` — Official RE-ARC Bench submission (125 tasks)
- `solves/{task_id}/solver.py` — 125 standalone Python solvers

## Combined TranscendPlexity Results

| Benchmark | Score | Note |
|-----------|-------|------|
| ARC-AGI-1 | 400/400 | Fixed evaluation set |
| ARC-AGI-2 | 120/120 | Curated hard tasks |
| ARC-AGI-3 | 20/20 | Interactive games |
| **RE-ARC** | **125/125** | **Procedural abstraction** |
| **TOTAL** | **665/665** | |

---

*TranscendPlexity demonstrates that program synthesis captures true abstract rules — not surface patterns. RE-ARC's procedural generation can't fool a solver that actually understands the transformation.*
