"""
ARC-AGI solver for task 2685904e

Rule:
- Row 0 has N cells of value 8 (left-aligned).
- Row 6 is a separator (all 5s).
- Row 8 is a "palette" of colored values.
- Count occurrences of each value in the palette. For each position where
  the palette value appears exactly N times, show that value; otherwise 0.
- Fill this pattern into the N rows directly above the separator (rows 6-N through 5).
"""
from collections import Counter


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    n_eights = sum(1 for v in grid[0] if v == 8)
    palette = grid[8]
    counts = Counter(palette)

    pattern = [v if counts[v] == n_eights else 0 for v in palette]

    for r in range(6 - n_eights, 6):
        out[r] = pattern[:]

    return out


if __name__ == "__main__":
    import json, pathlib

    task_path = pathlib.Path(__file__).resolve().parents[2] / "dataset" / "tasks" / "2685904e.json"
    with open(task_path) as f:
        task = json.load(f)

    all_pass = True
    for phase in ("train", "test"):
        for i, ex in enumerate(task[phase]):
            result = solve(ex["input"])
            ok = result == ex["output"]
            print(f"{phase}[{i}]: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r, (got, exp) in enumerate(zip(result, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}")
                        print(f"          exp {exp}")
    raise SystemExit(0 if all_pass else 1)
