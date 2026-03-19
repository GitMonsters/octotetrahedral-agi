from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    flat = [v for r in grid for v in r]
    bg = Counter(flat).most_common(1)[0][0]
    nonbg_colors = set(v for v in flat if v != bg)

    out = [[4] * W for _ in range(H)]

    if not nonbg_colors:
        for c in range(W):
            out[0][c] = 0
    elif 8 in nonbg_colors and bg != 8:
        for r in range(min(H, W)):
            out[r][r] = 0
    else:
        for r in range(min(H, W)):
            out[r][W - 1 - r] = 0

    return out
