"""
Solver for ARC task f45f5ca7.

Pattern: Each non-zero value in column 0 is moved to a fixed column
determined by the color:
  - 8 → column 1
  - 2 → column 2
  - 4 → column 3
  - 3 → column 4
The value is removed from column 0 and placed at its designated column
in the same row.
"""

import json
from typing import List

COLOR_TO_COL = {8: 1, 2: 2, 4: 3, 3: 4}


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        v = grid[r][0]
        if v != 0 and v in COLOR_TO_COL:
            out[r][COLOR_TO_COL[v]] = v
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/f45f5ca7.json") as f:
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
