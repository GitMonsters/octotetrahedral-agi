"""
Solver for ARC-AGI puzzle fe9372f3.

Rule: A plus/cross of 2s sits in the grid. From its center:
  - Diagonal lines of 1s extend in all four diagonal directions to grid edges.
  - Cardinal arms extend with repeating pattern [8, 8, 4] to grid edges.
"""

import json
import copy
from typing import List

TASK_PATH = "/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/fe9372f3.json"
EXTEND_PATTERN = [8, 8, 4]


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [[0] * cols for _ in range(rows)]

    # Find all 2-cells and compute the center of the cross
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    cr = sum(r for r, c in twos) // len(twos)
    cc = sum(c for r, c in twos) // len(twos)

    # Place the original cross
    for r, c in twos:
        out[r][c] = 2

    # Extend cardinal arms with repeating 8,8,4 pattern
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        # Start from the arm tip (one step from center in this direction)
        ar, ac = cr + dr, cc + dc
        # Walk from the arm tip outward
        r, c = ar + dr, ac + dc
        step = 0
        while 0 <= r < rows and 0 <= c < cols:
            out[r][c] = EXTEND_PATTERN[step % 3]
            r += dr
            c += dc
            step += 1

    # Extend diagonal lines of 1s from center
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        r, c = cr + dr, cc + dc
        while 0 <= r < rows and 0 <= c < cols:
            out[r][c] = 1
            r += dr
            c += dc

    return out


def main():
    with open(TASK_PATH) as f:
        task = json.load(f)

    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            print(f"Train {i}: FAIL")
            for r in range(len(expected)):
                if result[r] != expected[r]:
                    print(f"  Row {r}: got    {result[r]}")
                    print(f"          expect {expected[r]}")

    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        if "output" in ex:
            expected = ex["output"]
            if result == expected:
                print(f"Test  {i}: PASS")
            else:
                all_pass = False
                print(f"Test  {i}: FAIL")
                for r in range(len(expected)):
                    if result[r] != expected[r]:
                        print(f"  Row {r}: got    {result[r]}")
                        print(f"          expect {expected[r]}")
        else:
            print(f"Test  {i}: (no expected output)")
            for r in result:
                print(f"  {r}")

    if all_pass:
        print("\nAll examples PASS!")


if __name__ == "__main__":
    main()
