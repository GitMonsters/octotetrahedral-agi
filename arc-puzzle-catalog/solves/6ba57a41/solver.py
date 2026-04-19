from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    colors = Counter(v for row in grid for v in row)
    
    if len(colors) == 1:
        # Uniform grid
        bg = list(colors.keys())[0]
        r6 = H // 2
        w4 = max(1, W // 4)
        c6_start = (W - w4) // 2
        for c in range(c6_start, c6_start + w4):
            out[r6][c] = 6
        w8 = max(1, W // 8) + 1
        for c in range(W - w8, W):
            out[H - 1][c] = 7
        # 5 at top-left (may be invisible if bg==5)
        out[0][0] = 5
        return out
    
    # 2-color grid: find divider color (forms full rows AND cols)
    col_set = set(colors.keys())
    div_color = None
    for c_try in colors:
        row_divs = [r for r in range(H) if all(grid[r][c] == c_try for c in range(W))]
        col_divs = [c for c in range(W) if all(grid[r][c] == c_try for r in range(H))]
        if row_divs or col_divs:
            div_color = c_try
            break
    
    if div_color is None:
        return out
    
    row_divs_set = set(r for r in range(H) if all(grid[r][c] == div_color for c in range(W)))
    col_divs_set = set(c for c in range(W) if all(grid[r][c] == div_color for r in range(H)))
    
    # Find row segments
    row_segs = []
    i = 0
    while i < H:
        if i not in row_divs_set:
            j = i
            while j < H and j not in row_divs_set:
                j += 1
            row_segs.append((i, j - 1))
            i = j
        else:
            i += 1
    
    col_segs = []
    i = 0
    while i < W:
        if i not in col_divs_set:
            j = i
            while j < W and j not in col_divs_set:
                j += 1
            col_segs.append((i, j - 1))
            i = j
        else:
            i += 1
    
    N, M = len(row_segs), len(col_segs)
    if N == 0 or M == 0:
        return out
    
    def fill_cell(rs, cs, val):
        r1, r2 = rs
        c1, c2 = cs
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                out[r][c] = val
    
    fill_cell(row_segs[0], col_segs[0], 5)
    fill_cell(row_segs[N // 2], col_segs[M // 2], 6)
    fill_cell(row_segs[-1], col_segs[-1], 7)
    return out
