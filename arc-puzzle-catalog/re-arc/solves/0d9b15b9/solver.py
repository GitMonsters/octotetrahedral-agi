from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find background (most common color)
    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] += 1
    bg = color_count.most_common(1)[0][0]

    # Find horizontal band (full rows of non-bg color)
    band_color = None
    band_rows_set = set()
    for r in range(rows):
        if all(grid[r][c] == grid[r][0] for c in range(cols)) and grid[r][0] != bg:
            band_rows_set.add(r)
            band_color = grid[r][0]

    # Find vertical band (full columns of non-bg color) if no horizontal band
    band_cols_set = set()
    if not band_rows_set:
        for c in range(cols):
            col_vals = set(grid[r][c] for r in range(rows))
            if len(col_vals) == 1 and grid[0][c] != bg:
                band_cols_set.add(c)
                band_color = grid[0][c]

    # Find diagonal cells (non-bg, non-band)
    diag_cells = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg and (band_color is None or v != band_color):
                diag_cells.append((r, c))
    diag_cells.sort()

    if len(diag_cells) < 2:
        return out

    # Diagonal direction
    dr = 1 if diag_cells[1][0] > diag_cells[0][0] else (
        -1 if diag_cells[1][0] < diag_cells[0][0] else 0)
    dc = 1 if diag_cells[1][1] > diag_cells[0][1] else (
        -1 if diag_cells[1][1] < diag_cells[0][1] else 0)

    if band_rows_set:
        # Horizontal band: extend diagonal toward band, reflect off it
        band_min = min(band_rows_set)
        band_max = max(band_rows_set)

        d0 = min(abs(diag_cells[0][0] - band_min),
                 abs(diag_cells[0][0] - band_max))
        dN = min(abs(diag_cells[-1][0] - band_min),
                 abs(diag_cells[-1][0] - band_max))

        if dN <= d0:
            near_end = diag_cells[-1]
            ext_dr, ext_dc = dr, dc
        else:
            near_end = diag_cells[0]
            ext_dr, ext_dc = -dr, -dc

        if near_end[0] < band_min:
            gap = band_min - near_end[0] - 1
        else:
            gap = near_end[0] - band_max - 1

    elif band_cols_set:
        # Vertical band: extend diagonal toward band, reflect off it
        band_col_min = min(band_cols_set)
        band_col_max = max(band_cols_set)

        d0 = min(abs(diag_cells[0][1] - band_col_min),
                 abs(diag_cells[0][1] - band_col_max))
        dN = min(abs(diag_cells[-1][1] - band_col_min),
                 abs(diag_cells[-1][1] - band_col_max))

        if dN <= d0:
            near_end = diag_cells[-1]
            ext_dr, ext_dc = dr, dc
        else:
            near_end = diag_cells[0]
            ext_dr, ext_dc = -dr, -dc

        if near_end[1] < band_col_min:
            gap = band_col_min - near_end[1] - 1
        else:
            gap = near_end[1] - band_col_max - 1

    else:
        # No band: diagonal touches a grid edge, reflect using corner distance
        r0, c0 = diag_cells[0]
        rN, cN = diag_cells[-1]

        on_edge_0 = (r0 == 0 or r0 == rows - 1 or c0 == 0 or c0 == cols - 1)

        if on_edge_0:
            near_end = diag_cells[-1]
            ext_dr, ext_dc = dr, dc
            edge_r, edge_c = r0, c0
        else:
            near_end = diag_cells[0]
            ext_dr, ext_dc = -dr, -dc
            edge_r, edge_c = rN, cN

        if edge_r == 0 or edge_r == rows - 1:
            gap = min(edge_c, cols - 1 - edge_c)
        else:
            gap = min(edge_r, rows - 1 - edge_r)

    # Vertex position
    vr = near_end[0] + gap * ext_dr
    vc = near_end[1] + gap * ext_dc

    # Extension cells (between diagonal end and vertex)
    for t in range(1, gap + 1):
        r = near_end[0] + t * ext_dr
        c = near_end[1] + t * ext_dc
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = 6

    # Reflected arm direction
    if band_cols_set:
        refl_dr = ext_dr
        refl_dc = -ext_dc
    else:
        refl_dr = -ext_dr
        refl_dc = ext_dc

    t = 1
    while True:
        r = vr + t * refl_dr
        c = vc + t * refl_dc
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = 6
            t += 1
        else:
            break

    return out
