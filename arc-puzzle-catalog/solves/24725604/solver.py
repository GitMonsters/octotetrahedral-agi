"""Solver for ARC task 24725604.

Rule: The input is a grid with a dominant background color, a single solid
rectangle of another color, and scattered noise pixels. The output removes
all noise, keeping only the background and the rectangle.

Detection: find the largest axis-aligned solid rectangle of any non-background
color. Background = most frequent color.
"""

from collections import Counter
import numpy as np


def transform(grid_list: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_list, dtype=int)
    h, w = grid.shape

    bg = Counter(grid.flatten().tolist()).most_common(1)[0][0]

    best_area = 0
    best_rect = None

    for color in set(grid.flatten().tolist()) - {bg}:
        mask = (grid == color)
        # Use histogram approach per row to find max rectangle efficiently
        # but brute-force is fine for ARC grid sizes (≤30×30)
        for r1 in range(h):
            for c1 in range(w):
                if not mask[r1, c1]:
                    continue
                # Determine max width starting at (r1, c1)
                max_w = 0
                for c in range(c1, w):
                    if mask[r1, c]:
                        max_w = c - c1 + 1
                    else:
                        break
                # Extend downward, narrowing width as needed
                cur_w = max_w
                for r2 in range(r1, h):
                    if not mask[r2, c1]:
                        break
                    # Shrink width to fit this row
                    row_w = 0
                    for c in range(c1, c1 + cur_w):
                        if mask[r2, c]:
                            row_w = c - c1 + 1
                        else:
                            break
                    cur_w = row_w
                    if cur_w == 0:
                        break
                    area = (r2 - r1 + 1) * cur_w
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, r2, c1, c1 + cur_w - 1, color)

    out = np.full_like(grid, bg)
    if best_rect:
        r1, r2, c1, c2, color = best_rect
        out[r1:r2 + 1, c1:c2 + 1] = color

    return out.tolist()
