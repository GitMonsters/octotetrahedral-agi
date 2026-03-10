"""
Solver for ARC task f3e62deb.

Pattern: A 3x3 hollow square sits on a 10x10 grid. Based on the color,
slide the shape to the corresponding wall:
  - color 8 → right wall
  - color 4 → bottom wall
  - color 6 → top wall
  - color 3 → left wall
The shape keeps its position on the other axis.
"""

import json
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find the bounding box and color of the shape
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Extract the shape pattern relative to its bounding box
    shape = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        shape.append(row)

    # Determine new top-left position based on color
    if color == 8:    # move right
        new_r, new_c = min_r, cols - w
    elif color == 4:  # move down
        new_r, new_c = rows - h, min_c
    elif color == 6:  # move up
        new_r, new_c = 0, min_c
    elif color == 3:  # move left
        new_r, new_c = min_r, 0
    else:
        return grid

    out = [[0] * cols for _ in range(rows)]
    for dr in range(h):
        for dc in range(w):
            out[new_r + dr][new_c + dc] = shape[dr][dc]
    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/f3e62deb.json") as f:
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
