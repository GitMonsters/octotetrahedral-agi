"""
Solver for ARC task 46447928.

Rule: A small pattern with a 2x2 marker block is reflected 4-fold
(horizontal + vertical) around the center of the 2x2 marker block.

Detection:
- Background = most common color
- Marker = color with exactly 4 pixels forming a 2x2 block (if unique),
  otherwise the first 2x2 homogeneous non-bg block found
- All non-background pixels are reflected through the marker center
"""

import numpy as np
from collections import Counter


def transform(grid_input):
    grid = np.array(grid_input)
    H, W = grid.shape

    # Background = most common color
    vals, counts = np.unique(grid, return_counts=True)
    bg = vals[np.argmax(counts)]

    # Non-bg colors and their counts
    non_bg = {}
    for v, c in zip(vals, counts):
        if v != bg:
            non_bg[int(v)] = int(c)

    # Find the 2x2 marker block
    marker_pos = None

    # Strategy 1: a color with exactly 4 pixels forming a 2x2 block
    for color, count in non_bg.items():
        if count == 4:
            positions = list(zip(*np.where(grid == color)))
            rs = [p[0] for p in positions]
            cs = [p[1] for p in positions]
            if max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
                marker_pos = (min(rs), min(cs))
                break

    # Strategy 2: find any 2x2 homogeneous non-bg block
    if marker_pos is None:
        for r in range(H - 1):
            for c in range(W - 1):
                block = grid[r:r+2, c:c+2]
                if np.all(block == block[0, 0]) and block[0, 0] != bg:
                    marker_pos = (r, c)
                    break
            if marker_pos is not None:
                break

    # Center of 2x2 marker
    cy = marker_pos[0] + 0.5
    cx = marker_pos[1] + 0.5

    # Reflect all non-bg pixels through the center (4-fold symmetry)
    result = np.full_like(grid, bg)
    non_bg_positions = list(zip(*np.where(grid != bg)))

    for r, c in non_bg_positions:
        color = grid[r, c]
        reflections = [
            (r, c),
            (r, int(2 * cx - c)),
            (int(2 * cy - r), c),
            (int(2 * cy - r), int(2 * cx - c)),
        ]
        for rr, rc in reflections:
            if 0 <= rr < H and 0 <= rc < W:
                result[rr, rc] = color

    return result.tolist()
