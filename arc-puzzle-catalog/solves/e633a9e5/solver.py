"""
Solver for ARC task e633a9e5.

Transformation: 3x3 → 5x5 non-uniform scaling.
  - Rows 0 and 2 are doubled in height; row 1 stays single.
  - Columns 0 and 2 are doubled in width; column 1 stays single.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    row_map = [0, 0, 1, 2, 2]
    col_map = [0, 0, 1, 2, 2]
    return [[grid[row_map[r]][col_map[c]] for c in range(5)] for r in range(5)]


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/e633a9e5.json") as f:
        task = json.load(f)
    for i, pair in enumerate(task["train"] + task["test"]):
        result = solve(pair["input"])
        status = "PASS" if result == pair["output"] else "FAIL"
        print(f"Pair {i}: {status}")
