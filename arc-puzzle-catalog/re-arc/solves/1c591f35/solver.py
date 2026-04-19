from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find separator rows and separator color
    sep_rows = []
    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_rows.append(r)
            sep_color = grid[r][0]

    # Find separator columns
    sep_cols = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] == sep_color:
            sep_cols.append(c)

    # Row and column bands (sub-grid regions)
    row_bands = []
    start = 0
    for sr in sep_rows:
        if sr > start:
            row_bands.append((start, sr - 1))
        start = sr + 1
    if start < rows:
        row_bands.append((start, rows - 1))

    col_bands = []
    start = 0
    for sc in sep_cols:
        if sc > start:
            col_bands.append((start, sc - 1))
        start = sc + 1
    if start < cols:
        col_bands.append((start, cols - 1))

    # Background color (most common non-separator value)
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    non_sep = []
    for r in range(rows):
        if r in sep_row_set:
            continue
        for c in range(cols):
            if c in sep_col_set:
                continue
            non_sep.append(grid[r][c])
    bg_color = Counter(non_sep).most_common(1)[0][0]

    # Locate the single cell with color 6
    six_r, six_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6:
                six_r, six_c = r, c
                break
        if six_r is not None:
            break

    # Identify the sub-grid containing color 6
    winner_ri, winner_ci = None, None
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            if r0 <= six_r <= r1 and c0 <= six_c <= c1:
                winner_ri, winner_ci = ri, ci
                break

    # Local position of 6 within the winner sub-grid determines destination
    wr0, wr1 = row_bands[winner_ri]
    wc0, wc1 = col_bands[winner_ci]
    local_r = six_r - wr0
    local_c = six_c - wc0

    # Extract winner sub-grid content
    winner_content = []
    for r in range(wr0, wr1 + 1):
        winner_content.append([grid[r][c] for c in range(wc0, wc1 + 1)])

    # Build output: background + separators
    output = [[bg_color] * cols for _ in range(rows)]
    for sr in sep_rows:
        for c in range(cols):
            output[sr][c] = sep_color
    for sc in sep_cols:
        for r in range(rows):
            output[r][sc] = sep_color

    # Place winner content at the destination sub-grid
    dest_r0, dest_r1 = row_bands[local_r]
    dest_c0, dest_c1 = col_bands[local_c]
    for dr, sr in enumerate(range(dest_r0, dest_r1 + 1)):
        for dc, sc in enumerate(range(dest_c0, dest_c1 + 1)):
            output[sr][sc] = winner_content[dr][dc]

    return output
