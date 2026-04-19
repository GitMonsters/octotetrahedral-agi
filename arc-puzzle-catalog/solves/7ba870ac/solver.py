from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    cells = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
    mid = H // 2
    top = sorted([(r, c, v) for r, c, v in cells if r <= mid], key=lambda x: x[0])
    bot = sorted([(r, c, v) for r, c, v in cells if r > mid], key=lambda x: x[0])
    out = [[bg] * 3 for _ in range(3)]
    for i, (r, c, v) in enumerate(top):
        out[i][0] = v
    for i, (r, c, v) in enumerate(bot):
        out[2 - i][1] = v
    return out
