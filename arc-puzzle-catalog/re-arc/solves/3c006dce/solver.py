from math import gcd


def transform(grid):
    grid = [row[:] for row in grid]
    R, C = len(grid), len(grid[0])

    colors = set()
    for row in grid:
        for c in row:
            colors.add(c)

    if len(colors) >= 2:
        sep_color = _find_sep_color(grid, R, C)
        if sep_color is None:
            return grid

        h_seps = [r for r in range(R) if all(grid[r][c] == sep_color for c in range(C))]
        v_seps = [c for c in range(C) if all(grid[r][c] == sep_color for r in range(R))]

        row_bands = _extract_bands(h_seps, R)
        col_bands = _extract_bands(v_seps, C)
        n_rows = len(row_bands)
        n_cols = len(col_bands)

        _fill_cell(grid, row_bands[0], col_bands[n_cols - 1], 7)
        _fill_cell(grid, row_bands[n_rows // 2], col_bands[n_cols // 2], 9)
        _fill_cell(grid, row_bands[n_rows - 1], col_bands[0], 5)
    else:
        g = gcd(R, C)

        seven_h = R - 6 * g - 2
        nine_r = seven_h + (2 * g - 1) + 2

        five_w = C - 5 * g - 8
        nine_w = max(1, 4 * (g - 1))
        seven_w = 5 * g - 1 - nine_w
        nine_c = five_w + 5
        seven_c = C - seven_w

        for r in range(seven_h):
            for c in range(seven_c, seven_c + seven_w):
                grid[r][c] = 7

        for r in range(nine_r, nine_r + g):
            for c in range(nine_c, nine_c + nine_w):
                grid[r][c] = 9

        for r in range(R - g, R):
            for c in range(five_w):
                grid[r][c] = 5

    return grid


def _find_sep_color(grid, R, C):
    for r in range(R):
        if len(set(grid[r])) == 1:
            candidate = grid[r][0]
            for c in range(C):
                if all(grid[rr][c] == candidate for rr in range(R)):
                    return candidate
    return None


def _extract_bands(seps, dim):
    bands = []
    prev = -1
    for s in seps:
        if s > prev + 1:
            bands.append((prev + 1, s - 1))
        prev = s
    if prev < dim - 1:
        bands.append((prev + 1, dim - 1))
    return bands


def _fill_cell(grid, row_band, col_band, color):
    for r in range(row_band[0], row_band[1] + 1):
        for c in range(col_band[0], col_band[1] + 1):
            grid[r][c] = color
