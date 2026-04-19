"""
Solver for ARC task 45260785.

Rule: The input is a tiled/repeating pattern with a solid rectangular hole
filled with a single "hole color" (the unique color forming a solid rectangle).
The output is the content that should fill that hole — i.e., the continuation
of the tiling pattern.

Steps:
1. Detect the hole color — the color that forms a solid axis-aligned rectangle.
2. Locate the bounding box of that rectangle.
3. Determine the horizontal and vertical tiling periods of the pattern
   (ignoring the hole region).
4. Reconstruct the missing region by copying values from equivalent
   positions elsewhere in the tiled grid.
5. Return the reconstructed rectangle as the output.
"""

from typing import List
import numpy as np


def transform(grid_list: List[List[int]]) -> List[List[int]]:
    grid = np.array(grid_list, dtype=int)
    H, W = grid.shape

    # Step 1: Find the hole color — the color whose bounding box is entirely that color
    hole_color, r1, r2, c1, c2 = _find_hole(grid)

    # Step 2: Mask the hole region
    masked = grid.copy()
    masked[r1:r2 + 1, c1:c2 + 1] = -1

    # Step 3: Find tiling periods
    h_period = _find_period(masked, H, W, axis='h')
    v_period = _find_period(masked, H, W, axis='v')

    # Step 4: Reconstruct the hole
    rect_h = r2 - r1 + 1
    rect_w = c2 - c1 + 1
    result = np.zeros((rect_h, rect_w), dtype=int)

    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            result[r - r1, c - c1] = _lookup(masked, r, c, v_period, h_period, H, W)

    return result.tolist()


def _find_hole(grid: np.ndarray):
    """Find the color that forms a solid rectangle (the hole)."""
    H, W = grid.shape
    colors = np.unique(grid)
    for color in colors:
        rows, cols = np.where(grid == color)
        if len(rows) == 0:
            continue
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        rect_h = r2 - r1 + 1
        rect_w = c2 - c1 + 1
        # Must be a solid rectangle and match exact count
        if rect_h * rect_w == len(rows) and rect_h >= 2 and rect_w >= 2:
            region = grid[r1:r2 + 1, c1:c2 + 1]
            if np.all(region == color):
                return color, r1, r2, c1, c2
    raise ValueError("No solid rectangular hole found")


def _find_period(masked: np.ndarray, H: int, W: int, axis: str) -> int:
    """Find the smallest tiling period along the given axis."""
    if axis == 'h':
        for p in range(1, W):
            match = True
            for r in range(H):
                for c in range(W - p):
                    if masked[r, c] == -1 or masked[r, c + p] == -1:
                        continue
                    if masked[r, c] != masked[r, c + p]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return p
    else:  # vertical
        for p in range(1, H):
            match = True
            for r in range(H - p):
                for c in range(W):
                    if masked[r, c] == -1 or masked[r + p, c] == -1:
                        continue
                    if masked[r, c] != masked[r + p, c]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return p
    raise ValueError(f"No {axis} period found")


def _lookup(masked: np.ndarray, r: int, c: int,
            v_period: int, h_period: int, H: int, W: int) -> int:
    """Find the value at (r, c) by looking at equivalent tiled positions."""
    for dv in range(-((H // v_period) + 1), (H // v_period) + 2):
        for dh in range(-((W // h_period) + 1), (W // h_period) + 2):
            nr = r + dv * v_period
            nc = c + dh * h_period
            if 0 <= nr < H and 0 <= nc < W and masked[nr, nc] != -1:
                return int(masked[nr, nc])
    raise ValueError(f"Could not reconstruct cell ({r}, {c})")
