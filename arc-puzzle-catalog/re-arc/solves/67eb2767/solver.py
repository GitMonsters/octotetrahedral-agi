from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    markers = []
    mc = bg
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                markers.append((r, c))
                mc = grid[r][c]
    
    if not markers:
        return [row[:] for row in grid]
    
    # Use horizontal mode (bounce top/bottom) when grid is wider
    # Use vertical mode (bounce left/right) when grid is taller
    horizontal = W >= H
    
    result = [row[:] for row in grid]
    
    def trace_path(sr, sc, dr, dc):
        cells = []
        r, c = sr, sc
        for _ in range(H * W + 10):
            cells.append((r, c))
            nr, nc = r + dr, c + dc
            if horizontal:
                if nr < 0 or nr >= H:
                    dr = -dr
                    nr = r + dr
                if nc < 0 or nc >= W:
                    break
            else:
                if nc < 0 or nc >= W:
                    dc = -dc
                    nc = c + dc
                if nr < 0 or nr >= H:
                    break
            r, c = nr, nc
        return cells
    
    for mr, mrc in markers:
        if horizontal:
            dc = 1 if mrc == 0 else -1
            dr = 1 if mr == 0 else -1
        else:
            dr = 1 if mr == 0 else -1
            dc = -1 if mrc >= W // 2 else 1
            if mrc == 0:
                dc = 1
            elif mrc == W - 1:
                dc = -1
        
        path = trace_path(mr, mrc, dr, dc)
        for r, c in path:
            result[r][c] = mc
    
    return result
