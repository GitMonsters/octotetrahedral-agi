def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import copy
    from collections import Counter

    grid = copy.deepcopy(input_grid)
    rows = len(grid)
    cols = len(grid[0])

    # Background = most common color
    color_count = Counter(v for row in grid for v in row)
    bg = color_count.most_common(1)[0][0]

    # Detect wall: full rows or full columns of a single non-bg color
    wall_rows = set()
    wall_cols = set()
    wall_color = None

    for r in range(rows):
        if grid[r][0] != bg and all(grid[r][c] == grid[r][0] for c in range(cols)):
            wall_rows.add(r)
            wall_color = grid[r][0]

    for c in range(cols):
        if grid[0][c] != bg and all(grid[r][c] == grid[0][c] for r in range(rows)):
            wall_cols.add(c)
            wall_color = grid[0][c]

    wall_is_horizontal = len(wall_rows) > 0
    wall_is_vertical = len(wall_cols) > 0

    # Diagonal cells: non-bg, non-wall
    diag_cells = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v == bg:
                continue
            if wall_is_horizontal and r in wall_rows:
                continue
            if wall_is_vertical and c in wall_cols:
                continue
            diag_cells.append((r, c))

    diag_cells.sort()
    dr = diag_cells[1][0] - diag_cells[0][0]
    dc = diag_cells[1][1] - diag_cells[0][1]
    if dr != 0:
        dr = dr // abs(dr)
    if dc != 0:
        dc = dc // abs(dc)

    start_r, start_c = diag_cells[0]
    end_r, end_c = diag_cells[-1]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def is_wall(r: int, c: int) -> bool:
        if wall_is_horizontal and r in wall_rows:
            return True
        if wall_is_vertical and c in wall_cols:
            return True
        return False

    new_cells = []

    # Extend forward from end of diagonal
    r, c = end_r + dr, end_c + dc
    while in_bounds(r, c) and not is_wall(r, c):
        new_cells.append((r, c))
        r += dr
        c += dc

    if in_bounds(r, c) and is_wall(r, c):
        # Bounce off wall
        if wall_is_horizontal:
            b_dr, b_dc = -dr, dc
        else:
            b_dr, b_dc = dr, -dc
        last_r, last_c = r - dr, c - dc
        r2, c2 = last_r + b_dr, last_c + b_dc
        while in_bounds(r2, c2) and not is_wall(r2, c2):
            new_cells.append((r2, c2))
            r2 += b_dr
            c2 += b_dc

    # Extend backward from start of diagonal
    r, c = start_r - dr, start_c - dc
    while in_bounds(r, c) and not is_wall(r, c):
        new_cells.append((r, c))
        r -= dr
        c -= dc

    if in_bounds(r, c) and is_wall(r, c):
        # Bounce off wall (backward direction was -dr, -dc; flip appropriate component)
        if wall_is_horizontal:
            b_dr, b_dc = dr, -dc
        else:
            b_dr, b_dc = -dr, dc
        last_r, last_c = r + dr, c + dc
        r2, c2 = last_r + b_dr, last_c + b_dc
        while in_bounds(r2, c2) and not is_wall(r2, c2):
            new_cells.append((r2, c2))
            r2 += b_dr
            c2 += b_dc

    for r, c in new_cells:
        grid[r][c] = 0

    return grid
