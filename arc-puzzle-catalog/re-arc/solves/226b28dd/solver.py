def transform(grid):
    H, W = len(grid), len(grid[0])

    sep_val = None
    sep_rows = []
    for r in range(H):
        if len(set(grid[r])) == 1:
            sep_rows.append(r)
            sep_val = grid[r][0]

    sep_cols = []
    for c in range(W):
        col_vals = [grid[r][c] for r in range(H)]
        if len(set(col_vals)) == 1:
            sep_cols.append(c)

    row_bands = []
    prev = -1
    for r in sorted(sep_rows + [H]):
        if r - prev > 1:
            row_bands.append((prev + 1, r - 1))
        prev = r
    if prev < H - 1:
        row_bands.append((prev + 1, H - 1))

    col_bands = []
    prev = -1
    for c in sorted(sep_cols + [W]):
        if c - prev > 1:
            col_bands.append((prev + 1, c - 1))
        prev = c
    if prev < W - 1:
        col_bands.append((prev + 1, W - 1))

    output = []
    for r1, r2 in row_bands:
        row = []
        for c1, c2 in col_bands:
            colors = set()
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if grid[r][c] != sep_val:
                        colors.add(grid[r][c])
            if len(colors) == 1:
                row.append(colors.pop())
            else:
                row.append(sep_val)
        output.append(row)

    return output
