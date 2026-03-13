def solve(grid: list[list[int]]) -> list[list[int]]:
    """Recover the values hidden under the 7-rectangle using the grid's D4 symmetry.

    The grid has bilateral reflective symmetry about both a horizontal and
    vertical axis (both at the same position). A rectangular block of 7s masks
    some cells. The output is the original values at those 7-positions,
    recovered via the full dihedral-4 symmetry group (H-mirror, V-mirror,
    180° rotation, transpose, and their compositions).
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find 7-positions and their bounding box
    sevens = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 7:
                sevens.add((r, c))

    if not sevens:
        return grid

    min_r = min(r for r, _ in sevens)
    max_r = max(r for r, _ in sevens)
    min_c = min(c for _, c in sevens)
    max_c = max(c for _, c in sevens)

    # Detect the shared symmetry axis (axis_2 = 2 * axis)
    axis_2 = _find_axis(grid, rows, cols, sevens)

    # Build output by recovering each 7-cell
    out_rows = max_r - min_r + 1
    out_cols = max_c - min_c + 1
    result = [[0] * out_cols for _ in range(out_rows)]

    for r, c in sevens:
        mr = axis_2 - r
        mc = axis_2 - c
        # All 7 non-identity elements of the D4 symmetry group
        candidates = [
            (mr, c),    # horizontal reflection
            (r, mc),    # vertical reflection
            (mr, mc),   # 180° rotation
            (c, r),     # transpose
            (mc, r),    # transpose then V-reflect
            (c, mr),    # transpose then H-reflect
            (mc, mr),   # transpose then 180°
        ]
        val = 0
        for cr, cc in candidates:
            if 0 <= cr < rows and 0 <= cc < cols and (cr, cc) not in sevens:
                val = grid[cr][cc]
                break
        result[r - min_r][c - min_c] = val

    return result


def _find_axis(
    grid: list[list[int]], rows: int, cols: int, sevens: set[tuple[int, int]]
) -> int:
    """Detect the bilateral symmetry axis (returned as 2*axis, an integer)."""
    for axis_2 in range(rows - 1, rows + 6):
        ok = True
        checked = 0
        for r in range(rows):
            mr = axis_2 - r
            if not (0 <= mr < rows) or mr == r:
                continue
            for c in range(cols):
                if (r, c) in sevens or (mr, c) in sevens:
                    continue
                checked += 1
                if grid[r][c] != grid[mr][c]:
                    ok = False
                    break
            if not ok:
                break
        if ok and checked > 0:
            return axis_2
    return rows  # fallback: assume axis at N/2
