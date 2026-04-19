def transform(grid):
    import numpy as np
    from collections import deque, Counter
    
    grid = [list(row) for row in grid]
    inp = np.array(grid)
    R, C = inp.shape
    out = inp.copy()
    
    flat = [int(inp[r][c]) for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find connected components of non-bg cells
    visited = set()
    comps = []
    for r in range(R):
        for c in range(C):
            if int(inp[r][c]) != bg and (r,c) not in visited:
                q = deque([(r,c)])
                visited.add((r,c))
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc, int(inp[cr][cc])))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and int(inp[nr][nc]) != bg:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                comps.append(cells)
    
    # Process components with 8-cells (direction markers)
    for comp in comps:
        eight_cells = [(r,c) for r,c,v in comp if v == 8]
        non_eight = [(r,c,v) for r,c,v in comp if v != 8]
        non_eight_colors = set(v for _,_,v in non_eight)
        
        if not eight_cells or not non_eight:
            continue
        
        e_min_r = min(r for r,c in eight_cells)
        e_max_r = max(r for r,c in eight_cells)
        e_min_c = min(c for r,c in eight_cells)
        e_max_c = max(c for r,c in eight_cells)
        e_h = e_max_r - e_min_r + 1
        e_w = e_max_c - e_min_c + 1
        
        for color in non_eight_colors:
            cells_c = [(r,c) for r,c,v in comp if v == color]
            o_min_r = min(r for r,c in cells_c)
            o_max_r = max(r for r,c in cells_c)
            o_min_c = min(c for r,c in cells_c)
            o_max_c = max(c for r,c in cells_c)
            
            if e_min_c > o_max_c:  # 8 RIGHT
                t_right = e_min_c - 1
                t_left = t_right - 2 * e_w + 1
                t_top, t_bottom = e_min_r, e_max_r
            elif e_max_c < o_min_c:  # 8 LEFT
                t_left = e_max_c + 1
                t_right = t_left + 2 * e_w - 1
                t_top, t_bottom = e_min_r, e_max_r
            elif e_min_r > o_max_r:  # 8 BELOW
                t_bottom = e_min_r - 1
                t_top = t_bottom - 2 * e_h + 1
                t_left, t_right = e_min_c, e_max_c
            elif e_max_r < o_min_r:  # 8 ABOVE
                t_top = e_max_r + 1
                t_bottom = t_top + 2 * e_h - 1
                t_left, t_right = e_min_c, e_max_c
            else:
                continue
            
            for r in range(max(0, t_top), min(R, t_bottom + 1)):
                for c in range(max(0, t_left), min(C, t_right + 1)):
                    if int(out[r][c]) == bg:
                        out[r][c] = color
        
        # Fill 8-bbox
        for r in range(e_min_r, e_max_r + 1):
            for c in range(e_min_c, e_max_c + 1):
                if int(out[r][c]) == bg:
                    out[r][c] = 8
    
    # Process remaining components (no 8-cells): fill bbox per color
    # BUT: also consider nearby single-cell components of the same color
    # Strategy: for each non-8 color with >=2 cells in a component,
    # find the bounding box and fill
    for comp in comps:
        eight_cells = [(r,c) for r,c,v in comp if v == 8]
        if eight_cells:
            continue  # Already processed
        
        non_eight = [(r,c,v) for r,c,v in comp if v != 8]
        non_eight_colors = set(v for _,_,v in non_eight)
        
        for color in non_eight_colors:
            cells_c = [(r,c) for r,c,v in comp if v == color]
            if len(cells_c) < 2:
                continue
            
            min_r = min(r for r,c in cells_c)
            max_r = max(r for r,c in cells_c)
            min_c = min(c for r,c in cells_c)
            max_c = max(c for r,c in cells_c)
            
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if int(out[r][c]) == bg:
                        out[r][c] = color
    
    return out.tolist()
