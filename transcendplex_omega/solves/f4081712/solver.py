def solve(grid):
    H = len(grid)
    W = len(grid[0])
    # Find bounding box of all 3s (the "damaged" region)
    min_r, max_r = H, -1
    min_c, max_c = W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 3:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    # Recover values from vertically mirrored positions
    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[H - 1 - r][c])
        output.append(row)
    return output
