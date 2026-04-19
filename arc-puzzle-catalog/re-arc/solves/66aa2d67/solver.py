from collections import Counter, deque

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    # Find connected components of non-bg cells
    visited = [[False]*W for _ in range(H)]
    components = []
    
    def bfs(sr, sc):
        q = deque([(sr, sc)])
        visited[sr][sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c, grid[r][c]))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] != bg:
                    visited[nr][nc] = True
                    q.append((nr, nc))
        return cells
    
    for r in range(H):
        for c in range(W):
            if not visited[r][c] and grid[r][c] != bg:
                comp = bfs(r, c)
                components.append(comp)
    
    if not components:
        return [row[:] for row in grid]
    
    # Largest component is the big rect
    components.sort(key=len, reverse=True)
    big_comp = components[0]
    small_comps = components[1:]
    
    # Big rect bounding box and dominant color
    big_rows = [r for r,c,v in big_comp]
    big_cols = [c for r,c,v in big_comp]
    br0, br1 = min(big_rows), max(big_rows)
    bc0, bc1 = min(big_cols), max(big_cols)
    bh = br1 - br0 + 1
    bw = bc1 - bc0 + 1
    big_color = Counter(v for r,c,v in big_comp).most_common(1)[0][0]
    
    # Extract big rect local grid
    big_local = [[grid[br0+r][bc0+c] for c in range(bw)] for r in range(bh)]
    
    # Find cutouts (bg regions within the big rect bounding box)
    cutout_visited = [[False]*bw for _ in range(bh)]
    cutouts = []
    
    for r in range(bh):
        for c in range(bw):
            if big_local[r][c] == bg and not cutout_visited[r][c]:
                q = deque([(r, c)])
                cutout_visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < bh and 0 <= nc < bw and not cutout_visited[nr][nc] and big_local[nr][nc] == bg:
                            cutout_visited[nr][nc] = True
                            q.append((nr, nc))
                cutouts.append(cells)
    
    # Extract small shapes
    small_shapes = []
    for comp in small_comps:
        rows = [r for r,c,v in comp]
        cols = [c for r,c,v in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        sh = r1 - r0 + 1
        sw = c1 - c0 + 1
        shape = [[bg]*sw for _ in range(sh)]
        for r, c, v in comp:
            shape[r-r0][c-c0] = v
        small_shapes.append(shape)
    
    # Build output: big rect all filled with dominant color
    result = [[big_color]*bw for _ in range(bh)]
    
    # For each cutout, find matching small shape and paste it
    def get_rotations(shape):
        """Generate all 8 rotations/reflections"""
        shapes = []
        s = [row[:] for row in shape]
        for _ in range(4):
            shapes.append(s)
            # Also add horizontal flip
            shapes.append([row[::-1] for row in s])
            # Rotate 90° CW
            h, w = len(s), len(s[0])
            s = [[s[h-1-c][r] for c in range(h)] for r in range(w)]
        return shapes
    
    for cutout in cutouts:
        cr0 = min(r for r,c in cutout)
        cc0 = min(c for r,c in cutout)
        cr1 = max(r for r,c in cutout)
        cc1 = max(c for r,c in cutout)
        ch = cr1 - cr0 + 1
        cw = cc1 - cc0 + 1
        
        # Cutout mask within its bounding box
        cutout_set = set((r-cr0, c-cc0) for r,c in cutout)
        
        # Find matching small shape
        matched = False
        for si, shape in enumerate(small_shapes):
            for rot in get_rotations(shape):
                rh, rw = len(rot), len(rot[0])
                if rh != ch or rw != cw:
                    continue
                # Check if dominant-color cells match cutout
                dom_cells = set()
                for r in range(rh):
                    for c in range(rw):
                        if rot[r][c] == big_color:
                            dom_cells.add((r, c))
                        elif rot[r][c] == bg:
                            dom_cells.add((r, c))
                
                # Check if non-bg, dominant-color cells match cutout
                shape_dom = set()
                for r in range(rh):
                    for c in range(rw):
                        if rot[r][c] == big_color:
                            shape_dom.add((r, c))
                
                if shape_dom == cutout_set:
                    # Paste the rotated shape at cutout position
                    for r in range(rh):
                        for c in range(rw):
                            result[cr0+r][cc0+c] = rot[r][c] if rot[r][c] != bg else big_color
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            # No match: just fill cutout with big color (already done)
            pass
    
    return result
