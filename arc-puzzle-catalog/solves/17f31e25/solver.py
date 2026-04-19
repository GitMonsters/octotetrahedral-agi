def transform(grid: list[list[int]]) -> list[list[int]]:
    H = len(grid)
    W = len(grid[0])
    max_ring = min(H, W) // 2

    # Extract color of each concentric ring
    ring_colors = [grid[r][r] for r in range(max_ring)]

    # Cyclic right-shift: innermost becomes outermost
    shifted = [ring_colors[-1]] + ring_colors[:-1]

    # Build output by assigning each cell the shifted color for its ring
    out = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            ring = min(r, c, H - 1 - r, W - 1 - c)
            out[r][c] = shifted[ring]
    return out
