"""Solver for ARC-AGI task b7999b51.

Pattern: Each non-zero color forms a shape. Compute each color's bounding-box
height (number of rows it spans). Output is a "bar chart" — one column per
color, sorted tallest-first, filled from the top with that color.
"""

import json
from typing import List

Grid = List[List[int]]


def solve(grid: Grid) -> Grid:
    # Find row span (bounding-box height) for each color
    color_rows: dict[int, tuple[int, int]] = {}
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != 0:
                if val not in color_rows:
                    color_rows[val] = (r, r)
                else:
                    lo, hi = color_rows[val]
                    color_rows[val] = (min(lo, r), max(hi, r))

    # height = max_row - min_row + 1
    color_heights = [(hi - lo + 1, color) for color, (lo, hi) in color_rows.items()]
    # Sort descending by height, break ties by color value (arbitrary but consistent)
    color_heights.sort(key=lambda x: (-x[0], x[1]))

    max_h = color_heights[0][0]
    n_cols = len(color_heights)
    out = [[0] * n_cols for _ in range(max_h)]
    for col_idx, (h, color) in enumerate(color_heights):
        for row_idx in range(h):
            out[row_idx][col_idx] = color
    return out


if __name__ == "__main__":
    task = json.load(open("/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/b7999b51.json"))
    ok = True
    for i, p in enumerate(task["train"]):
        result = solve(p["input"])
        match = result == p["output"]
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            ok = False
            print(f"  Expected: {p['output']}")
            print(f"  Got:      {result}")
    for i, p in enumerate(task["test"]):
        result = solve(p["input"])
        print(f"Test {i} output: {result}")
    if ok:
        print("\nAll training examples PASS!")
