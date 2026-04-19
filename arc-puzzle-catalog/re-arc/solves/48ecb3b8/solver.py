def transform(grid):
    R, C = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    markers = [(r, c, grid[r][c]) for r in range(R) for c in range(C) if grid[r][c] != bg]
    out = [row[:] for row in grid]
    hw = 2
    
    if len(markers) == 2:
        (r1, c1, col1), (r2, c2, col2) = sorted(markers)
        c = c1
        arm = (r2 - r1 - 1) // 2
        for r in range(r1 + 1, r1 + arm - 1):
            out[r][c] = col1
        cb = r1 + arm - 1
        for cc in range(c - hw, c + hw + 1):
            if 0 <= cc < C: out[cb][cc] = col1
        ext = r1 + arm
        for cc in [c - hw, c + hw]:
            if 0 <= cc < C: out[ext][cc] = col1
        for r in range(r2 - 1, r2 - arm + 1, -1):
            out[r][c] = col2
        cb = r2 - arm + 1
        for cc in range(c - hw, c + hw + 1):
            if 0 <= cc < C: out[cb][cc] = col2
        ext = r2 - arm
        for cc in [c - hw, c + hw]:
            if 0 <= cc < C: out[ext][cc] = col2
    elif len(markers) == 1:
        r, c, col = markers[0]
        arm = 2 * hw + 1
        for rr in range(r - 1, r - arm + 1, -1):
            out[rr][c] = col
        cb = r - arm + 1
        for cc in range(c - hw, c + hw + 1):
            if 0 <= cc < C: out[cb][cc] = col
        ext = r - arm
        for cc in [c - hw, c + hw]:
            if 0 <= cc < C: out[ext][cc] = col
    return out
