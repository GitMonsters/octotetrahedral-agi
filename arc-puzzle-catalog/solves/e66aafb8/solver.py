def solve(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])

    # Find the bounding box of the 0-region
    min_r, max_r, min_c, max_c = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Fill from the vertically mirrored position (top-bottom symmetry)
    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[H - 1 - r][c])
        output.append(row)
    return output
