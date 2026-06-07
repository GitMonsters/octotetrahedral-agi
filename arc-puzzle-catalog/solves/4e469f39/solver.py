"""
Task 4e469f39 solver.

Rule: The grid contains shapes of color 5 forming almost-closed rectangles.
Each shape has exactly one gap cell in its top row. The transformation:
  1. Fills the enclosed interior (including the gap cell) with color 2.
  2. Draws a horizontal line of 2s in the row immediately above the shape,
     extending from the gap column to the grid edge in the direction that has
     more 5-cells in the top row of the shape (left if tie).

Background = most frequent color. Wall color = the other color present in input.
Fill color = 2 (fixed transformation color for this task, not present in input).
"""

from collections import Counter
from typing import List


def solve(grid: List[List[int]]) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])

    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    wall_color = next(v for v in set(flat) if v != bg)
    fill_color = 2

    visited: set = set()

    def bfs_wall(sr: int, sc: int) -> List[tuple]:
        q, cells = [(sr, sc)], []
        visited.add((sr, sc))
        while q:
            r, c = q.pop(0)
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr][nc] == wall_color
                ):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return cells

    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == wall_color and (r, c) not in visited:
                components.append(bfs_wall(r, c))

    out = [row[:] for row in grid]

    for comp in components:
        top_row = min(r for r, c in comp)
        left_col = min(c for r, c in comp)
        right_col = max(c for r, c in comp)

        top_cells_cols = {c for r, c in comp if r == top_row}
        gap_col = next(c for c in range(left_col, right_col + 1) if c not in top_cells_cols)

        left_count = sum(1 for c in range(left_col, gap_col) if c in top_cells_cols)
        right_count = sum(1 for c in range(gap_col + 1, right_col + 1) if c in top_cells_cols)
        go_left = left_count >= right_count

        # Flood-fill interior from the gap cell (only downward + sideways, not upward)
        interior: set = set()
        q = [(top_row, gap_col)]
        interior.add((top_row, gap_col))
        while q:
            r, c = q.pop(0)
            for dr, dc in ((0, -1), (0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in interior
                    and grid[nr][nc] == bg
                ):
                    interior.add((nr, nc))
                    q.append((nr, nc))

        for r, c in interior:
            out[r][c] = fill_color

        line_row = top_row - 1
        if line_row >= 0:
            if go_left:
                for c in range(0, gap_col + 1):
                    out[line_row][c] = fill_color
            else:
                for c in range(gap_col, cols):
                    out[line_row][c] = fill_color

    return out
