def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])
    cr = (H - 1) // 2
    cc = (W - 1) // 2
    bg = input_grid[0][0]

    # Find non-background points grouped by color
    colors = {}
    for r in range(H):
        for c in range(W):
            v = input_grid[r][c]
            if v != bg:
                colors.setdefault(v, []).append((abs(r - cr), abs(c - cc)))

    output = [[bg] * W for _ in range(H)]

    for color, pts in colors.items():
        if len(pts) == 1:
            # Single point: 4-way reflection
            dr, dc = pts[0]
            for sr in (-1, 1):
                for sc in (-1, 1):
                    r, c = cr + sr * dr, cc + sc * dc
                    if 0 <= r < H and 0 <= c < W:
                        output[r][c] = color
        elif len(pts) == 2:
            (r1, c1), (r2, c2) = pts
            turning = (r1 < r2 and c1 > c2) or (r1 > r2 and c1 < c2)

            if turning:
                max_r, min_r = max(r1, r2), min(r1, r2)
                max_c, min_c = max(c1, c2), min(c1, c2)
                # Horizontal edges at rows ±max_r, cols from -min_c to +min_c
                for sr in (-1, 1):
                    row = cr + sr * max_r
                    if 0 <= row < H:
                        for dc in range(-min_c, min_c + 1, 2):
                            col = cc + dc
                            if 0 <= col < W:
                                output[row][col] = color
                # Vertical edges at cols ±max_c, rows from -min_r to +min_r
                for sc in (-1, 1):
                    col = cc + sc * max_c
                    if 0 <= col < W:
                        for dr in range(-min_r, min_r + 1, 2):
                            row = cr + dr
                            if 0 <= row < H:
                                output[row][col] = color
            else:
                # Non-turning: just reflect both points
                for dr, dc in pts:
                    for sr in (-1, 1):
                        for sc in (-1, 1):
                            r, c = cr + sr * dr, cc + sc * dc
                            if 0 <= r < H and 0 <= c < W:
                                output[r][c] = color

    return output
