"""
Solver for ARC task f3cdc58f.

Pattern: Count occurrences of colors 1-4 in the input grid, then build a
bar chart from the bottom-left. Column 0 gets 1s, column 1 gets 2s,
column 2 gets 3s, column 3 gets 4s. Each bar's height equals the count
of that color in the input.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v in counts:
                counts[v] += 1

    out = [[0] * cols for _ in range(rows)]
    for color, col_idx in [(1, 0), (2, 1), (3, 2), (4, 3)]:
        h = counts[color]
        for r in range(rows - h, rows):
            out[r][col_idx] = color
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/f3cdc58f.json") as f:
        task = json.load(f)

    all_pass = True
    for split in ["train", "test"]:
        for i, ex in enumerate(task[split]):
            result = solve(ex["input"])
            ok = result == ex["output"]
            print(f"{split} {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_pass = False
                for r, (got, exp) in enumerate(zip(result, ex["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}, expected {exp}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
