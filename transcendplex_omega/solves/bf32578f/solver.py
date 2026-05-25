"""
ARC-AGI Task bf32578f — Bracket Reflection Solver
Rule: For each colored shape, find max_c (rightmost column = back wall).
  - Rows containing max_c: erase (fill 0).
  - Other rows: find c_inner = max column in that row (closest to max_c).
    Fill segment [c_inner+1, 2*max_c - c_inner] with the color.
Verified 4/4 (3 train + 1 test ground truth).
"""
from collections import defaultdict
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    out = [[0] * W for _ in range(H)]

    color_cells: dict = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r, c))

    for color, cells in color_cells.items():
        max_c = max(c for _, c in cells)

        rows: dict = defaultdict(list)
        for r, c in cells:
            rows[r].append(c)

        for r, cols in rows.items():
            if max_c in cols:
                # Erase this row (back-wall row)
                continue
            # Fill from inner edge to its mirror about max_c
            c_inner = max(cols)
            mirror = 2 * max_c - c_inner
            for c in range(c_inner + 1, mirror + 1):
                if 0 <= c < W:
                    out[r][c] = color

    return out
