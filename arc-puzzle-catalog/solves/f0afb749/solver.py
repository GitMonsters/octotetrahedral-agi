"""Solver for ARC-AGI puzzle f0afb749.

Pattern: Each non-zero entry (i,j) in the NxN input defines a circular shift
k = (j - i) % N. For every unique shift k, a full diagonal of 2x2 blocks is
placed in the 2N x 2N output: colored [[v,v],[v,v]] where input is non-zero,
identity [[1,0],[0,1]] where input is zero.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    n = len(grid)
    out = [[0] * (2 * n) for _ in range(2 * n)]

    # Collect unique shifts from non-zero entries
    shifts = set()
    for i in range(n):
        for j in range(n):
            if grid[i][j] != 0:
                shifts.add((j - i) % n)

    # For each shift, lay down a diagonal of 2x2 blocks
    for k in shifts:
        for i in range(n):
            j = (i + k) % n
            r, c = 2 * i, 2 * j
            v = grid[i][j]
            if v != 0:
                out[r][c] = out[r][c + 1] = v
                out[r + 1][c] = out[r + 1][c + 1] = v
            else:
                out[r][c] = 1
                out[r + 1][c + 1] = 1

    return out


if __name__ == "__main__":
    with open("/tmp/arc_task_f0afb749.json") as f:
        task = json.load(f)

    ok = True
    for idx, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Train {idx}: PASS")
        else:
            print(f"Train {idx}: FAIL")
            ok = False

    for idx, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        expected = ex.get("output")
        if expected and result == expected:
            print(f"Test  {idx}: PASS")
        elif expected:
            print(f"Test  {idx}: FAIL")
            ok = False
        else:
            print(f"Test  {idx}: (no expected output)")

    print("\nAll correct!" if ok else "\nSome failures.")
