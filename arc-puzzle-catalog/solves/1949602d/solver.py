def transform(grid):
    from collections import Counter
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]

    non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not non_bg:
        return [row[:] for row in grid]

    bb_r1 = min(r for r, c in non_bg)
    bb_r2 = max(r for r, c in non_bg)
    bb_c1 = min(c for r, c in non_bg)
    bb_c2 = max(c for r, c in non_bg)

    if bb_r1 <= rows - 1 - bb_r2:
        ext_r1, ext_r2 = 0, bb_r2
    else:
        ext_r1, ext_r2 = bb_r1, rows - 1

    if bb_c1 <= cols - 1 - bb_c2:
        ext_c1, ext_c2 = 0, bb_c2
    else:
        ext_c1, ext_c2 = bb_c1, cols - 1

    tile_h = ext_r2 - ext_r1 + 1
    tile_w = ext_c2 - ext_c1 + 1
    period_v = tile_h + 1
    period_h = tile_w + 1

    out = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            dr = (r - ext_r1) % period_v
            dc = (c - ext_c1) % period_h
            if dr < tile_h and dc < tile_w:
                out[r][c] = grid[ext_r1 + dr][ext_c1 + dc]
    return out
