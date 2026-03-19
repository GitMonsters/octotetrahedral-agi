def transform(grid):
    H = len(grid)
    W = len(grid[0])
    from collections import Counter
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]

    min_r, max_r, min_c, max_c = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    cr = (min_r + max_r) // 2
    cc = (min_c + max_c) // 2
    fill = grid[cr][cc]

    output = [row[:] for row in grid]

    corners_dirs = [
        (min_r, min_c, -1, -1),
        (min_r, max_c, -1, +1),
        (max_r, min_c, +1, -1),
        (max_r, max_c, +1, +1),
    ]

    for r0, c0, dr, dc in corners_dirs:
        r, c = r0 + dr, c0 + dc
        while 0 <= r < H and 0 <= c < W:
            output[r][c] = fill
            r += dr
            c += dc

    return output
