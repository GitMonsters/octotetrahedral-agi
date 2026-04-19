def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    # Find background (most frequent)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find source cells
    sources = []
    color = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                sources.append((r, c))
                color = grid[r][c]
    
    if not sources:
        return [[0]*W for _ in range(H)]
    
    def bounce(x, L):
        if L == 1:
            return 0
        p = 2 * (L - 1)
        x = x % p
        if x >= L:
            x = p - x
        return x
    
    num_steps = max(H, W) - 1
    
    out = [[0]*W for _ in range(H)]
    
    for sr, sc in sources:
        # Determine direction toward interior
        if sr == 0:
            dr = 1
        elif sr == H - 1:
            dr = -1
        else:
            dr = 1 if sr < H // 2 else -1
        
        if sc == 0:
            dc = 1
        elif sc == W - 1:
            dc = -1
        else:
            dc = 1 if sc < W // 2 else -1
        
        for t in range(num_steps + 1):
            r = bounce(sr + dr * t, H)
            c = bounce(sc + dc * t, W)
            out[r][c] = color
    
    return out
