"""
Solver for ARC-AGI task d47aa2ff.

Rule: Input is a 10x21 grid split into left/right 10x10 halves by a column of 5s.
The right half is a near-copy of the left with some cells shifted.
- Where left has a value but right has 0 → mark as 2 (source / moved FROM)
- Where left has 0 but right has a value → mark as 1 (destination / moved TO)
- Otherwise keep the left value.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    rows = len(grid)
    cols = (len(grid[0]) - 1) // 2

    left = [row[:cols] for row in grid]
    right = [row[cols + 1:] for row in grid]

    output = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            l, ri = left[r][c], right[r][c]
            if l != 0 and ri == 0:
                output[r][c] = 2  # source: cell moved away
            elif l == 0 and ri != 0:
                output[r][c] = 1  # destination: cell moved here
            else:
                output[r][c] = l  # unchanged
    return output


if __name__ == "__main__":
    with open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/d47aa2ff.json") as f:
        task = json.load(f)

    # Verify against all training examples
    all_pass = True
    for i, ex in enumerate(task["train"]):
        result = solve(ex["input"])
        expected = ex["output"]
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"  Mismatch at ({r},{c}): got {result[r][c]}, expected {expected[r][c]}")

    # Solve test examples
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {i} output:")
        for row in result:
            print(" ".join(str(c) for c in row))

    print(f"\nAll training examples pass: {all_pass}")
