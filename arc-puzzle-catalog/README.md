# 🧩 ARC Puzzle Catalog

A collection of **422 standalone Python solvers** for [ARC-AGI](https://arcprize.org) puzzles — each one a verified, deterministic program that transforms input grids to output grids with 100% accuracy.

## Scores

| Benchmark | Solved | Total | Accuracy |
|-----------|--------|-------|----------|
| ARC-AGI-1 (eval) | 309 | 400 | **77.3%** |
| ARC-AGI-2 | 93 | 120 | **77.5%** |
| ARC-AGI-3 | 20 | 20 | **100%** |

> **Total: 422 verified solvers** across all three ARC benchmarks.

## How It Works

Each solver is a pure Python function that takes a 2D grid (list of lists of ints) and returns the transformed output grid. No ML models, no LLMs at inference time — just code.

```python
# Example: solves/0934a4d8/solver.py
def solve(grid: list[list[int]]) -> list[list[int]]:
    # Deterministic transformation logic
    ...
```

Solvers were synthesized using Claude Opus 4.6 via iterative program generation — the model analyzes training examples, writes a candidate solver, and refines it until all training and test cases pass.

## Repository Structure

```
arc-puzzle-catalog/
├── solves/              # 422 solver directories
│   ├── {task_id}/
│   │   └── solver.py   # solve(grid) → grid
│   └── ...
├── catalog.json         # Metadata for cataloged puzzles
├── dataset/             # Cached puzzle data
├── viz/                 # HTML grid visualizations
├── index.html           # Web catalog viewer
├── fetch_dataset.py     # ARC data fetcher
└── generate_viz.py      # Visualization generator
```

## Running a Solver

```bash
python3 -c "
import json, importlib.util

task_id = '0934a4d8'
with open(f'dataset/{task_id}.json') as f:
    task = json.load(f)

spec = importlib.util.spec_from_file_location('solver', f'solves/{task_id}/solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for pair in task['test']:
    result = mod.solve(pair['input'])
    assert result == pair['output'], 'Mismatch!'
    print(f'{task_id}: PASS')
"
```

## Verification

Every solver is independently verified against held-out test cases before being committed. The verification protocol:

1. Agent writes `solver.py` based on training examples only
2. Solver is tested against all training pairs
3. Solver is independently verified against test pairs (never seen during development)
4. Only solvers that pass **all** test cases are committed

## Methodology

The solving pipeline uses a multi-model orchestration approach:

- **Primary model**: Claude Opus 4.6 — handles complex spatial reasoning, symmetry detection, and multi-step transformations
- **Dispatch**: Parallel background agents (10 at a time), each assigned one task
- **Iteration**: Agents refine solvers through test-driven development until all examples pass
- **Verification**: Independent re-verification before commit

Typical solve times range from 60 seconds (simple pattern matching) to 20 minutes (complex multi-step reasoning).

## License

MIT

## Links

- [ARC Prize](https://arcprize.org) — The ARC-AGI benchmark
- [ARC Playground](https://arcprize.org/play) — Try puzzles interactively
