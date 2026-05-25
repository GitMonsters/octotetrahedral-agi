"""
ARC-AGI Task c35c1b4c — Horizontal Mirror Fill Solver
Rule: Find the dominant (most common non-zero) color.
For every cell (r, c): if the dominant color appears at (r, c) OR at (r, W-1-c),
  set output[r][c] = dominant_color.  Otherwise keep the original value.
Effectively reflects each row of dominant-color cells about the horizontal
center axis (col (W-1)/2), filling in the mirror image.
Verified 4/4 (3 train + 1 test ground truth).
"""
from collections import Counter
from typing import List


def find_dominant(grid: List[List[int]]) -> int:
    counts: Counter = Counter(v for row in grid for v in row if v != 0)
    return counts.most_common(1)[0][0]


def solve(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    dominant = find_dominant(grid)
    out = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == dominant or grid[r][W - 1 - c] == dominant:
                out[r][c] = dominant
    return out
