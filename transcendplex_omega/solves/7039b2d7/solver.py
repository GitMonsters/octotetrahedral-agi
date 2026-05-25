def solve(grid):
    """Condensed grid: find separator lines, output (row_bands x col_bands) of background color."""
    rows = len(grid)
    cols = len(grid[0])

    # Find the separator color: color that forms at least one complete row AND one complete column
    from collections import Counter
    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_color = grid[r][0]
            break
    if sep_color is None:
        # Fallback: find column that's all same
        for c in range(cols):
            col_vals = [grid[r][c] for r in range(rows)]
            if len(set(col_vals)) == 1:
                sep_color = col_vals[0]
                break

    # Find separator rows (all cells = sep_color)
    sep_rows = set()
    for r in range(rows):
        if all(grid[r][c] == sep_color for c in range(cols)):
            sep_rows.add(r)

    # Find separator columns (all cells = sep_color)
    sep_cols = set()
    for c in range(cols):
        if all(grid[r][c] == sep_color for r in range(rows)):
            sep_cols.add(c)

    # Compute row bands (contiguous non-separator rows)
    row_bands = []
    band = []
    for r in range(rows):
        if r in sep_rows:
            if band:
                row_bands.append(band)
                band = []
        else:
            band.append(r)
    if band:
        row_bands.append(band)

    # Compute col bands
    col_bands = []
    band = []
    for c in range(cols):
        if c in sep_cols:
            if band:
                col_bands.append(band)
                band = []
        else:
            band.append(c)
    if band:
        col_bands.append(band)

    # Background color: most common non-separator color
    counts = Counter()
    for r in range(rows):
        if r in sep_rows:
            continue
        for c in range(cols):
            if c in sep_cols:
                continue
            counts[grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]

    # Output: condensed grid
    out = [[bg] * len(col_bands) for _ in range(len(row_bands))]
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
