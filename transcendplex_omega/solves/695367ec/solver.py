"""
Solver for ARC task 695367ec.

The input is a uniform NxN grid of a single color C.
The output is a 15x15 grid where grid lines (every N+1 cells) are filled
with color C, and all other cells are 0.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    N = len(grid)
    color = grid[0][0]
    step = N + 1  # grid line spacing

    out = [[0] * 15 for _ in range(15)]
    for r in range(15):
        for c in range(15):
            if r % step == N or c % step == N:
                out[r][c] = color
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/695367ec.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                for r_idx, (got, exp) in enumerate(zip(result, expected)):
                    if got != exp:
                        print(f"  row {r_idx}: got {got}")
                        print(f"  row {r_idx}: exp {exp}")
