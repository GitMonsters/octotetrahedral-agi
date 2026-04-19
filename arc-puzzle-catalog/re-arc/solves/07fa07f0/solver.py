def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])

    # Find background color (from uniform rows or columns)
    bg = None
    for r in range(H):
        if len(set(input_grid[r])) == 1:
            bg = input_grid[r][0]
            break
    if bg is None:
        for c in range(W):
            col_vals = [input_grid[r][c] for r in range(H)]
            if len(set(col_vals)) == 1:
                bg = col_vals[0]
                break

    # Find pattern region bounds (non-background cells)
    min_r, max_r = H, -1
    min_c, max_c = W, -1
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Find tile row period
    tile_h = None
    for period in range(1, max_r - min_r + 2):
        match = True
        for r in range(min_r, max_r + 1 - period):
            for c in range(min_c, max_c + 1):
                if input_grid[r][c] != input_grid[r + period][c]:
                    match = False
                    break
            if not match:
                break
        if match:
            tile_h = period
            break

    # Find tile column period
    tile_w = None
    for period in range(1, max_c - min_c + 2):
        match = True
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1 - period):
                if input_grid[r][c] != input_grid[r][c + period]:
                    match = False
                    break
            if not match:
                break
        if match:
            tile_w = period
            break

    # Extract tile from pattern origin
    tile = []
    for r in range(tile_h):
        row = []
        for c in range(tile_w):
            row.append(input_grid[min_r + r][min_c + c])
        tile.append(row)

    # Compute phase shift based on pattern origin position
    r0, c0 = min_r, min_c
    row_shift = ((-r0) % tile_w + (-c0) % tile_h) % tile_h
    col_shift = 0

    # Generate output: tile the entire grid with the shifted phase
    output = []
    for r in range(H):
        row = []
        for c in range(W):
            row.append(tile[(r + row_shift) % tile_h][(c + col_shift) % tile_w])
        output.append(row)

    return output
