"""
ARC-AGI solver for task ca8de6ea.

The 5×5 input has non-zero values on its two diagonals forming an X shape
(9 unique cells). The output is a 3×3 grid that compresses the X by mapping:
  - Outer corners → output corners
  - Inner diagonal cells → output edges
  - Center → output center
"""
import json


def solve(grid: list[list[int]]) -> list[list[int]]:
    return [
        [grid[0][0], grid[1][1], grid[0][4]],
        [grid[3][1], grid[2][2], grid[1][3]],
        [grid[4][0], grid[3][3], grid[4][4]],
    ]


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/ca8de6ea.json") as f:
        task = json.load(f)

    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            result = solve(ex["input"])
            status = "PASS" if result == ex["output"] else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                print(f"  expected: {ex['output']}")
                print(f"  got:      {result}")
