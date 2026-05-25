"""
Solver for ARC-AGI puzzle b4a43f3b

Rule:
  Input is split by a row of 5s into two 6×6 grids.
  - Top grid: 3×3 arrangement of 2×2 color blocks → a 3×3 "stamp" palette.
  - Bottom grid: template where 2s mark positions on a 6×6 canvas.
  - Output is 18×18 (each template cell → 3×3 output block).
  - For every template cell == 2, stamp the 3×3 palette into the output.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    # Find the separator row (all 5s)
    sep = next(r for r, row in enumerate(grid) if all(v == 5 for v in row))

    # Extract 3×3 palette from the top 6×6 (2×2 blocks → single cells)
    palette = [[grid[r * 2][c * 2] for c in range(3)] for r in range(3)]

    # Extract 6×6 template from below the separator
    template = [grid[sep + 1 + r] for r in range(6)]

    # Build 18×18 output
    out = [[0] * 18 for _ in range(18)]
    for tr in range(6):
        for tc in range(6):
            if template[tr][tc] == 2:
                for pr in range(3):
                    for pc in range(3):
                        out[tr * 3 + pr][tc * 3 + pc] = palette[pr][pc]
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b4a43f3b.json") as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        if result == ex["output"]:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            all_pass = False
            for r, (got, exp) in enumerate(zip(result, ex["output"])):
                if got != exp:
                    print(f"  Row {r}: got  {got}")
                    print(f"          want {exp}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex and result == ex["output"]:
            print(f"Test  {i}: PASS")
        elif "output" in ex:
            print(f"Test  {i}: FAIL")
            all_pass = False
        else:
            print(f"Test  {i}: (no expected output to verify)")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
