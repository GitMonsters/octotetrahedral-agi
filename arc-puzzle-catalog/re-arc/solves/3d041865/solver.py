"""Solver for ARC puzzle 3d041865.

Rule: Scattered pixels shoot beams toward nearest wall lines.
- Walls are full rows/columns of a single non-background color.
- Each pixel shoots toward the nearest wall in each direction.
- Beam fills pixel color from pixel to 2 cells before wall.
- Wall ±1 positions get 3-cell perpendicular bars of wall color.
- Wall intersection gets pixel color.
"""
from collections import Counter


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in input_grid]
    R, C = len(grid), len(grid[0])

    # Background = most common color
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find wall rows/cols (entirely one non-bg color)
    wall_color = None
    wall_rows: set[int] = set()
    wall_cols: set[int] = set()

    for r in range(R):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            wall_rows.add(r)
            wall_color = grid[r][0]

    for c in range(C):
        col_vals = set(grid[r][c] for r in range(R))
        if len(col_vals) == 1 and grid[0][c] != bg:
            wall_cols.add(c)
            wall_color = grid[0][c]

    # Find scattered pixels (non-bg, not on wall row/col)
    pixels = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and r not in wall_rows and c not in wall_cols:
                pixels.append((r, c, grid[r][c]))

    for pr, pc, pcolor in pixels:
        # Shoot toward nearest wall columns (left and right)
        if wall_cols:
            left = [wc for wc in wall_cols if wc < pc]
            if left:
                _shoot_h(grid, R, C, pr, pc, pcolor, max(left), wall_color)
            right = [wc for wc in wall_cols if wc > pc]
            if right:
                _shoot_h(grid, R, C, pr, pc, pcolor, min(right), wall_color)

        # Shoot toward nearest wall rows (up and down)
        if wall_rows:
            above = [wr for wr in wall_rows if wr < pr]
            if above:
                _shoot_v(grid, R, C, pr, pc, pcolor, max(above), wall_color)
            below = [wr for wr in wall_rows if wr > pr]
            if below:
                _shoot_v(grid, R, C, pr, pc, pcolor, min(below), wall_color)

    return grid


def _shoot_h(grid, R, C, pr, pc, pcolor, wc, wcolor):
    """Shoot horizontal beam from pixel at (pr, pc) toward wall column wc."""
    # Beam: pixel color from pixel to 2 cells before wall
    if pc < wc:
        for c in range(pc, wc - 1):
            grid[pr][c] = pcolor
    else:
        for c in range(wc + 2, pc + 1):
            grid[pr][c] = pcolor

    # Expansion bars at wc-1 and wc+1: 3 cells vertically (pr-1, pr, pr+1)
    for delta in [-1, 1]:
        ec = wc + delta
        if 0 <= ec < C:
            for dr in [-1, 0, 1]:
                er = pr + dr
                if 0 <= er < R:
                    grid[er][ec] = wcolor

    # Wall intersection gets pixel color
    grid[pr][wc] = pcolor


def _shoot_v(grid, R, C, pr, pc, pcolor, wr, wcolor):
    """Shoot vertical beam from pixel at (pr, pc) toward wall row wr."""
    # Beam: pixel color from pixel to 2 cells before wall
    if pr < wr:
        for r in range(pr, wr - 1):
            grid[r][pc] = pcolor
    else:
        for r in range(wr + 2, pr + 1):
            grid[r][pc] = pcolor

    # Expansion bars at wr-1 and wr+1: 3 cells horizontally (pc-1, pc, pc+1)
    for delta in [-1, 1]:
        er = wr + delta
        if 0 <= er < R:
            for dc in [-1, 0, 1]:
                ec = pc + dc
                if 0 <= ec < C:
                    grid[er][ec] = wcolor

    # Wall intersection gets pixel color
    grid[wr][pc] = pcolor
