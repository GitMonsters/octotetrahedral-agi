"""
Solver for ARC task c1990cce.

Transformation: 1×N → N×N.
A single '2' in the input row spawns a V-shaped pattern of 2's expanding
diagonally, then interior/continuing cells are filled with 1's based on
modular diagonal conditions.

For cell (r, c) with p = column of the 2:
  a = r + c - p   (distance along right-going diagonal from V left arm)
  b = r - c + p   (distance along left-going diagonal from V right arm)
  - color 2 if on V arms: (a == 0 and b >= 0) or (b == 0 and a >= 0)
  - color 1 if inside/below V: a > 0 and b > 0 and b % 4 == 0
  - else 0
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    row = grid[0]
    n = len(row)
    p = row.index(2)
    output = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            a = r + c - p
            b = r - c + p
            if (a == 0 and b >= 0) or (b == 0 and a >= 0):
                output[r][c] = 2
            elif a > 0 and b > 0 and b % 4 == 0:
                output[r][c] = 1
    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/c1990cce.json") as f:
        task = json.load(f)
    for i, pair in enumerate(task["train"] + task["test"]):
        result = solve(pair["input"])
        status = "PASS" if result == pair["output"] else "FAIL"
        print(f"Pair {i}: {status}")
