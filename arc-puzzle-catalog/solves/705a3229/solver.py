"""
ARC-AGI Puzzle 705a3229 Solver

Rule: Each non-zero pixel draws an L-shape toward its nearest grid corner
(by Manhattan distance). One arm extends along the row to the corner's
column edge, the other along the column to the corner's row edge.
"""

import json
import copy
from typing import List

Grid = List[List[int]]

TASK_PATH = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/705a3229.json"


def solve(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    out = copy.deepcopy(grid)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                continue
            val = grid[r][c]

            corners = [
                (0, 0),            # top-left
                (0, cols - 1),     # top-right
                (rows - 1, 0),    # bottom-left
                (rows - 1, cols - 1),  # bottom-right
            ]
            # Nearest corner by Manhattan distance
            cr, cc = min(corners, key=lambda corner: abs(corner[0] - r) + abs(corner[1] - c))

            # Vertical arm toward corner row
            step = 1 if cr >= r else -1
            for rr in range(r, cr + step, step):
                out[rr][c] = val

            # Horizontal arm toward corner col
            step = 1 if cc >= c else -1
            for cc2 in range(c, cc + step, step):
                out[r][cc2] = val

    return out


if __name__ == "__main__":
    with open(TASK_PATH) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        match = result == ex["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] != ex["output"][r][c]:
                        print(f"  diff at ({r},{c}): got {result[r][c]}, expected {ex['output'][r][c]}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            match = result == ex["output"]
            print(f"Test  {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                all_pass = False
        else:
            print(f"Test  {i}: (no answer key)")
            for row in result:
                print(row)

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
