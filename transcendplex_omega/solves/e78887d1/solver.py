"""
ARC-AGI Puzzle e78887d1 Solver

Rule: The grid contains R row-groups and N color-columns of 3x3 patterns,
separated by zero-rows/columns. The patterns follow a cyclic permutation:
  pattern(row_r, col_c) = cycle[(r + c) % N]
where cycle[] is read from the first row-group.

The output is the next row in the sequence:
  output(col_c) = cycle[(R + c) % N]
colored with each column's color.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    H, W = len(grid), len(grid[0])
    R = (H - 1) // 4  # number of input row-groups
    N = (W + 1) // 4  # number of color-columns

    # Read cycle patterns and colors from the first row-group
    cycle = []
    colors = []
    for c in range(N):
        r0, c0 = 1, 4 * c
        block = [
            [1 if grid[r][cc] > 0 else 0 for cc in range(c0, c0 + 3)]
            for r in range(r0, r0 + 3)
        ]
        cycle.append(block)
        colors.append(max(grid[r][cc] for r in range(r0, r0 + 3) for cc in range(c0, c0 + 3)))

    # Build output: 3 rows wide, same column structure as input
    output = []
    for row in range(3):
        out_row = []
        for c in range(N):
            pat = cycle[(R + c) % N]
            color = colors[c]
            for j in range(3):
                out_row.append(color * pat[row][j])
            if c < N - 1:
                out_row.append(0)  # separator column
        output.append(out_row)

    return output


if __name__ == "__main__":
    import os

    task_path = os.path.join(
        os.path.expanduser("~"),
        "ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/e78887d1.json",
    )
    with open(task_path) as f:
        task = json.load(f)

    # Verify all training examples
    all_pass = True
    for i, ex in enumerate(task["train"]):
        predicted = solve(ex["input"])
        expected = ex["output"]
        match = predicted == expected
        status = "PASS ✓" if match else "FAIL ✗"
        print(f"Train {i}: {status}")
        if not match:
            all_pass = False
            print(f"  Expected: {expected}")
            print(f"  Got:      {predicted}")

    # Solve test cases
    for i, ex in enumerate(task["test"]):
        result = solve(ex["input"])
        print(f"\nTest {i} solution: {result}")
        if "output" in ex:
            match = result == ex["output"]
            print(f"  Matches expected: {match}")

    print(f"\n{'ALL TRAINING EXAMPLES PASSED' if all_pass else 'SOME EXAMPLES FAILED'}")
