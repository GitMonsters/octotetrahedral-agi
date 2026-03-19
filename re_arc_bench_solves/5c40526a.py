def transform(grid):
    H = len(grid)
    W = len(grid[0])

    min_r, max_r, min_c, max_c = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    px = None
    for r in range(H):
        if 8 not in grid[r]:
            row = grid[r]
            for p in range(1, W + 1):
                if all(row[c] == row[c % p] for c in range(W)):
                    px = p
                    break
            break

    py = None
    for c in range(W):
        col = [grid[r][c] for r in range(H)]
        if 8 not in col:
            for p in range(1, H + 1):
                if all(col[r] == col[r % p] for r in range(H)):
                    py = p
                    break
            break

    tile = [[None] * px for _ in range(py)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8:
                tile[r % py][c % px] = grid[r][c]

    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(tile[r % py][c % px])
        output.append(row)

    return output
