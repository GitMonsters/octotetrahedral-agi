def transform(grid):
    H = len(grid)
    W = len(grid[0])
    from collections import Counter
    
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common()[0][0]
    out = [row[:] for row in grid]
    
    # Each non-bg dot is replaced with bg and gets 4 diagonal neighbors:
    # UL(-1,-1), UR(-1,+1), DL(+1,-1) → color 2
    # DR(+1,+1) → color 1
    REGULAR = 2
    SPECIAL = 1
    
    dots = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                dots.append((r, c))
    
    for r, c in dots:
        out[r][c] = bg
        for dr, dc, color in [(-1,-1,REGULAR), (-1,+1,REGULAR), (+1,-1,REGULAR), (+1,+1,SPECIAL)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = color
    
    return out
