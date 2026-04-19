def transform(grid):
    H, W = len(grid), len(grid[0])
    n = min(H, W) // 2
    ring_colors = [grid[k][k] for k in range(n)]
    shifted = [ring_colors[-1]] + ring_colors[:-1]
    if n >= 2 and shifted[0] == shifted[-1]:
        wrapping_color = shifted[0]
        if wrapping_color not in shifted[1:-1]:
            bands = []
            for k, c in enumerate(ring_colors):
                if not bands or bands[-1][0] != c:
                    bands.append([c, 1])
                else:
                    bands[-1][1] += 1
            if len(bands) >= 2:
                band_colors = [b[0] for b in bands]
                shifted_colors = [band_colors[-1]] + band_colors[:-1]
                inner_width = bands[-1][1]
                result = [shifted_colors[0]]
                remaining = n - 1 - inner_width
                n_middle = len(bands) - 2
                if n_middle > 0:
                    per_mid = remaining // n_middle
                    extra = remaining % n_middle
                    for i in range(1, len(bands) - 1):
                        w = per_mid + (1 if i - 1 < extra else 0)
                        result.extend([shifted_colors[i]] * w)
                result.extend([shifted_colors[-1]] * inner_width)
                shifted = result
    output = [[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            ring = min(r, c, H-1-r, W-1-c)
            output[r][c] = shifted[ring]
    return output
