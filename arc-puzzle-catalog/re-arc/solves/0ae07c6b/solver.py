def transform(input_grid):
    """
    The grid has separator lines (one color) dividing it into rectangular cells.
    The transformation colors three specific cells:
    - Top-right cell (row_band 0, col_band last) → color 5
    - Center cell (row_band N//2, col_band M//2) → color 5
    - Bottom-left cell (row_band last, col_band 0) → color 9
    """
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find separator color (the color that forms full rows)
    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_color = grid[r][0]
            break

    # Find horizontal separator rows
    h_seps = [r for r in range(rows) if all(grid[r][c] == sep_color for c in range(cols))]

    # Find vertical separator columns
    v_seps = [c for c in range(cols) if all(grid[r][c] == sep_color for r in range(rows))]

    # Extract row bands (contiguous non-separator row groups)
    row_bands = []
    prev = -1
    for r in h_seps:
        if r > prev + 1:
            row_bands.append((prev + 1, r - 1))
        prev = r
    if prev < rows - 1:
        row_bands.append((prev + 1, rows - 1))

    # Extract col bands
    col_bands = []
    prev = -1
    for c in v_seps:
        if c > prev + 1:
            col_bands.append((prev + 1, c - 1))
        prev = c
    if prev < cols - 1:
        col_bands.append((prev + 1, cols - 1))

    N = len(row_bands)
    M = len(col_bands)

    # Color three cells on the anti-diagonal
    targets = [
        (0, M - 1, 5),         # top-right → gray
        (N // 2, M // 2, 5),   # center → gray
        (N - 1, 0, 9),         # bottom-left → maroon
    ]

    for ri, ci, color in targets:
        r_start, r_end = row_bands[ri]
        c_start, c_end = col_bands[ci]
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                grid[r][c] = color

    return grid
