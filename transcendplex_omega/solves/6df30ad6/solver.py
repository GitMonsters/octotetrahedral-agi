"""
ARC-AGI solver for task 6df30ad6

Transformation: A shape made of 5s exists in the grid, along with scattered
colored dots (non-0, non-5). The dot whose Euclidean distance to the nearest
5-cell is smallest determines the recolor: all 5s become that dot's color,
and everything else becomes 0.
"""
import json
import math
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Collect 5-cells and scattered dots
    five_cells = []
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_cells.append((r, c))
            elif grid[r][c] != 0:
                dots.append((r, c, grid[r][c]))

    # Find the dot closest (Euclidean) to any 5-cell
    best_dist = float("inf")
    best_color = 0
    for dr, dc, color in dots:
        for fr, fc in five_cells:
            dist = math.hypot(dr - fr, dc - fc)
            if dist < best_dist:
                best_dist = dist
                best_color = color

    # Output: 5-shape recolored, everything else cleared
    out = [[0] * cols for _ in range(rows)]
    for r, c in five_cells:
        out[r][c] = best_color

    return out


if __name__ == "__main__":
    with open("/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/6df30ad6.json") as f:
        task = json.load(f)

    all_pass = True
    for split in ("train", "test"):
        for i, example in enumerate(task[split]):
            result = solve(example["input"])
            expected = example["output"]
            status = "PASS" if result == expected else "FAIL"
            print(f"{split}[{i}]: {status}")
            if status == "FAIL":
                all_pass = False
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
