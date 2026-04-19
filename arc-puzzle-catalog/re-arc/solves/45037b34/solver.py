from collections import Counter

def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])

    # Find vertical symmetry breaks (top half vs bottom half)
    vert_break_pairs = []
    for r in range(H // 2):
        mr = H - 1 - r
        for c in range(W):
            if grid[r][c] != grid[mr][c]:
                vert_break_pairs.append((r, c, grid[r][c], grid[mr][c]))

    top_vals = [v for _, _, v, _ in vert_break_pairs]
    bot_vals = [mv for _, _, _, mv in vert_break_pairs]
    top_unique = len(set(top_vals))
    bot_unique = len(set(bot_vals))

    candidates = []
    if top_unique == 1:
        paint_color = top_vals[0]
        cells = {(r, c) for r, c, _, _ in vert_break_pairs}
        candidates.append((paint_color, cells))
    if bot_unique == 1:
        paint_color = bot_vals[0]
        cells = {(H - 1 - r, c) for r, c, _, _ in vert_break_pairs}
        candidates.append((paint_color, cells))

    best = None
    best_area = -1
    for paint_color, cells in candidates:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        # Expand rectangle while all border cells match paint color
        while min_r > 0 and all(grid[min_r - 1][c] == paint_color for c in range(min_c, max_c + 1)):
            min_r -= 1
        while max_r < H - 1 and all(grid[max_r + 1][c] == paint_color for c in range(min_c, max_c + 1)):
            max_r += 1
        while min_c > 0 and all(grid[r][min_c - 1] == paint_color for r in range(min_r, max_r + 1)):
            min_c -= 1
        while max_c < W - 1 and all(grid[r][max_c + 1] == paint_color for r in range(min_r, max_r + 1)):
            max_c += 1
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        if area > best_area:
            best_area = area
            best = (paint_color, min_r, max_r, min_c, max_c)

    paint_color, r1, r2, c1, c2 = best

    # Reconstruct hidden content using 4-fold symmetry mirrors
    result = []
    for r in range(r1, r2 + 1):
        row = []
        for c in range(c1, c2 + 1):
            mr, mc = H - 1 - r, W - 1 - c
            if not (r1 <= mr <= r2 and c1 <= c <= c2):
                row.append(grid[mr][c])
            elif not (r1 <= r <= r2 and c1 <= mc <= c2):
                row.append(grid[r][mc])
            elif not (r1 <= mr <= r2 and c1 <= mc <= c2):
                row.append(grid[mr][mc])
            else:
                row.append(paint_color)
        result.append(row)
    return result
