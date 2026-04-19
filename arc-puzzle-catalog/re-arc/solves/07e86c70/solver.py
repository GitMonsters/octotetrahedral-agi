def transform(grid):
    H, W = len(grid), len(grid[0])
    bg_count = sum(1 for r in grid for v in r if v == 5)
    non_bg_count = H * W - bg_count
    M = bg_count
    OH, OW = M * H, M * W
    out = [[5] * OW for _ in range(OH)]
    placed = 0
    for br in range(M - 1, -1, -1):
        for bc in range(M):
            if placed >= non_bg_count:
                break
            for r in range(H):
                for c in range(W):
                    out[br * H + r][bc * W + c] = grid[r][c]
            placed += 1
        if placed >= non_bg_count:
            break
    return out
