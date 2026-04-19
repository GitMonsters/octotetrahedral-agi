def transform(grid):
    grid = [list(row) for row in grid]
    R = len(grid)
    C = len(grid[0])
    
    # Find background
    from collections import Counter
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]
    
    out = [row[:] for row in grid]
    
    # Find all 2x2 blocks of non-bg cells
    blocks = []
    for r in range(R - 1):
        for c in range(C - 1):
            if (grid[r][c] != bg and grid[r][c+1] != bg and 
                grid[r+1][c] != bg and grid[r+1][c+1] != bg):
                blocks.append((r, c))
    
    # For each block, project in 4 diagonal directions
    for br, bc in blocks:
        tl = grid[br][bc]
        tr = grid[br][bc+1]
        bl = grid[br+1][bc]
        brr = grid[br+1][bc+1]
        
        # NW: color = BR (opposite corner), position = (br-2, bc-2)
        projections = [
            (-2, -2, brr),   # NW -> BR color
            (-2, +2, bl),    # NE -> BL color
            (+2, -2, tr),    # SW -> TR color
            (+2, +2, tl),    # SE -> TL color
        ]
        
        for dr, dc, color in projections:
            for pr in range(2):
                for pc in range(2):
                    nr = br + dr + pr
                    nc = bc + dc + pc
                    if 0 <= nr < R and 0 <= nc < C and out[nr][nc] == bg:
                        out[nr][nc] = color
    
    return out
