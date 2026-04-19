from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    counts = Counter(c for r in grid for c in r)
    bg = counts.most_common(1)[0][0]
    
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                min_r = min(min_r, r); max_r = max(max_r, r)
                min_c = min(min_c, c); max_c = max(max_c, c)
    if max_r == -1:
        return out
    
    pat_h = max_r - min_r + 1
    pat_w = max_c - min_c + 1
    pattern = [[grid[r][c] for c in range(min_c, max_c+1)] for r in range(min_r, max_r+1)]
    
    def find_vp(pat):
        h, w = len(pat), len(pat[0])
        for p in range(1, h):
            if all(pat[r][c] == pat[r%p][c] for r in range(h) for c in range(w)):
                return p
        return h
    def find_hp(pat):
        h, w = len(pat), len(pat[0])
        for p in range(1, w):
            if all(pat[r][c] == pat[r][c%p] for r in range(h) for c in range(w)):
                return p
        return w
    
    vp, hp = find_vp(pattern), find_hp(pattern)
    bg_rows, bg_cols = rows - pat_h, cols - pat_w
    
    if bg_rows >= bg_cols and bg_rows > 0:
        for r in range(rows):
            pr = ((r - min_r) % vp + vp) % vp
            for c in range(min_c, max_c + 1):
                out[r][c] = pattern[pr][c - min_c]
    elif bg_cols > bg_rows and bg_cols > 0:
        for r in range(min_r, max_r + 1):
            for c in range(cols):
                pc = ((c - min_c) % hp + hp) % hp
                out[r][c] = pattern[r - min_r][pc]
    return out
