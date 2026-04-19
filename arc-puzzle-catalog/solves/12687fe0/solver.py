from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Separator = most common color overall
    flat = [c for row in grid for c in row]
    sep = Counter(flat).most_common(1)[0][0]

    # Find full separator rows and columns
    sep_rows = [r for r in range(H) if all(grid[r][c] == sep for c in range(W))]
    sep_cols = [c for c in range(W) if all(grid[r][c] == sep for r in range(H))]

    # Extract row bands (ranges between separator rows)
    row_bands = []
    prev = 0
    for sr in sep_rows:
        if sr > prev:
            row_bands.append((prev, sr - 1))
        prev = sr + 1
    if prev < H:
        row_bands.append((prev, H - 1))

    # Extract column bands (ranges between separator columns)
    col_bands = []
    prev = 0
    for sc in sep_cols:
        if sc > prev:
            col_bands.append((prev, sc - 1))
        prev = sc + 1
    if prev < W:
        col_bands.append((prev, W - 1))

    # Build output: one cell per panel
    output = []
    for r0, r1 in row_bands:
        row = []
        for c0, c1 in col_bands:
            # Collect non-separator colors in this panel
            colors = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if grid[r][c] != sep:
                        colors.append(grid[r][c])
            if colors:
                row.append(Counter(colors).most_common(1)[0][0])
            else:
                row.append(sep)
        output.append(row)

    return output
