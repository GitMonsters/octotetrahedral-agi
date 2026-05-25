"""Solver for ARC task ce039d91.

Rule: For each cell with value 5, check the horizontally mirrored position
(col' = width - 1 - col) in the same row. If that position also has a 5,
change this cell to 1. Otherwise keep it as 5.
"""

from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                mirror_c = cols - 1 - c
                if grid[r][mirror_c] == 5:
                    result[r][c] = 1
    return result


if __name__ == "__main__":
    import json

    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/ce039d91.json") as f:
        task = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            output = solve(example["input"])
            if output == example["output"]:
                print(f"{split} {i}: PASS")
            else:
                print(f"{split} {i}: FAIL")
                all_pass = False
                for r, (got, exp) in enumerate(zip(output, example["output"])):
                    if got != exp:
                        print(f"  row {r}: got {got}")
                        print(f"          exp {exp}")

    if all_pass:
        print("All examples passed!")
