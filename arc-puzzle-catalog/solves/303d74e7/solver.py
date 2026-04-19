"""Solver for ARC task 303d74e7.

Rule: Find all maximal solid rectangles (≥3×3) of the dominant color
(most common), then draw a 1-pixel black (0) border around each one.
"""

import numpy as np
from collections import Counter


def transform(grid_in):
    grid = np.array(grid_in)
    out = grid.copy()
    H, W = grid.shape

    dominant = Counter(grid.flatten().tolist()).most_common(1)[0][0]
    dom = (grid == dominant)

    # For each cell, compute consecutive run of dominant color to the right
    right_run = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W - 1, -1, -1):
            if dom[r, c]:
                right_run[r, c] = (right_run[r, c + 1] if c + 1 < W else 0) + 1
            else:
                right_run[r, c] = 0

    # Enumerate all solid rectangles of dominant color (≥3×3)
    rects = set()
    for r1 in range(H):
        for c1 in range(W):
            if right_run[r1, c1] < 3:
                continue
            min_width = right_run[r1, c1]
            for r2 in range(r1, H):
                min_width = min(min_width, right_run[r2, c1])
                if min_width < 3:
                    break
                if r2 - r1 + 1 >= 3:
                    rects.add((r1, c1, r2, c1 + min_width - 1))

    # Keep only maximal rectangles (not contained in any strictly larger one)
    rects = list(rects)
    maximal = []
    for i, (r1, c1, r2, c2) in enumerate(rects):
        contained = False
        for j, (r1b, c1b, r2b, c2b) in enumerate(rects):
            if i != j and r1b <= r1 and c1b <= c1 and r2b >= r2 and c2b >= c2 and \
               (r1b < r1 or c1b < c1 or r2b > r2 or c2b > c2):
                contained = True
                break
        if not contained:
            maximal.append((r1, c1, r2, c2))

    # Draw 1-pixel black border around each maximal rectangle
    for r1, c1, r2, c2 in maximal:
        br1, bc1 = max(0, r1 - 1), max(0, c1 - 1)
        br2, bc2 = min(H - 1, r2 + 1), min(W - 1, c2 + 1)
        for c in range(bc1, bc2 + 1):
            out[br1, c] = 0
            out[br2, c] = 0
        for r in range(br1, br2 + 1):
            out[r, bc1] = 0
            out[r, bc2] = 0

    return out.tolist()
