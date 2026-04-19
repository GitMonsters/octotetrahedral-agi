def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    from collections import Counter
    colors = Counter(v for row in grid for v in row)
    bg = colors.most_common(1)[0][0]
    
    def find_rect(r1, c1):
        B = grid[r1][c1]
        if B == bg: return None
        # Find right edge (all B along row r1)
        c2 = c1 + 1
        while c2 < W and grid[r1][c2] == B: c2 += 1
        c2 -= 1
        if c2 == c1: return None
        # Find bottom edge (all B along col c1)
        r2 = r1 + 1
        while r2 < H and grid[r2][c1] == B: r2 += 1
        r2 -= 1
        if r2 == r1: return None
        # Verify rectangle borders
        if not all(grid[r1][c] == B for c in range(c1, c2+1)): return None
        if not all(grid[r2][c] == B for c in range(c1, c2+1)): return None
        if not all(grid[r][c1] == B for r in range(r1, r2+1)): return None
        if not all(grid[r][c2] == B for r in range(r1, r2+1)): return None
        if r2 - r1 < 2 or c2 - c1 < 2: return None
        # Check interior is uniform
        I = grid[r1+1][c1+1]
        for r in range(r1+1, r2):
            for c in range(c1+1, c2):
                if grid[r][c] != I: return None
        return r1, c1, r2, c2, B, I
    
    found_cells = set()
    rects = []
    for r in range(H):
        for c in range(W):
            if (r, c) in found_cells or grid[r][c] == bg: continue
            res = find_rect(r, c)
            if res:
                r1, c1, r2, c2, B, I = res
                h = r2 - r1 - 1
                w = c2 - c1 - 1
                rects.append((r1, c1, r2, c2, B, I, h, w))
                for rr in range(r1, r2+1):
                    for cc in range(c1, c2+1):
                        found_cells.add((rr, cc))
    
    for r1, c1, r2, c2, B, I, h, w in rects:
        # Swap B and I within original rectangle (only if different)
        if B != I:
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if out[r][c] == B: out[r][c] = I
                    elif out[r][c] == I: out[r][c] = B
        
        # Add new outer border of color B (extending h rows top/bot, w cols left/right)
        for dr in range(1, h+1):
            for c in range(c1, c2+1):
                if 0 <= r1-dr < H: out[r1-dr][c] = B
                if 0 <= r2+dr < H: out[r2+dr][c] = B
        for dc in range(1, w+1):
            for r in range(r1, r2+1):
                if 0 <= c1-dc < W: out[r][c1-dc] = B
                if 0 <= c2+dc < W: out[r][c2+dc] = B
    
    return out
