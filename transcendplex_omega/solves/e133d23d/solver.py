"""
Solver for ARC task e133d23d.

Rule: The input is a 3x7 grid split by a column of 4s (column 3) into
two 3x3 regions. The output is the logical OR of the two regions:
if either region has a non-zero value at a position, output 2; else 0.
"""

import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    out = [[0] * 3 for _ in range(rows)]
    for r in range(rows):
        for c in range(3):
            left = grid[r][c]
            right = grid[r][c + 4]
            out[r][c] = 2 if (left != 0 or right != 0) else 0
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/e133d23d.json") as f:
        task = json.load(f)

    for idx, example in enumerate(task["train"]):
        result = solve(example["input"])
        expected = example["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Train {idx}: {status}")
        if result != expected:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    for idx, example in enumerate(task["test"]):
        result = solve(example["input"])
        expected = example["output"]
        status = "PASS" if result == expected else "FAIL"
        print(f"Test  {idx}: {status}")
        if result != expected:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
