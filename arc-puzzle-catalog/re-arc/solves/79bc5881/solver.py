from collections import Counter

def transform(grid):
    R, C = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(R) for c in range(C)).most_common(1)[0][0]
    out = [row[:] for row in grid]
    
    # Find connected components (4-connected) of non-bg
    nonbg = {(r,c): grid[r][c] for r in range(R) for c in range(C) if grid[r][c] != bg}
    visited = set()
    components = []
    for (r,c) in nonbg:
        if (r,c) in visited: continue
        queue = [(r,c)]
        visited.add((r,c))
        cells = [(r,c)]
        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if (nr,nc) in nonbg and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
                    cells.append((nr,nc))
        colors = set(nonbg[p] for p in cells)
        components.append((sorted(cells), colors))
    
    # Find multi-color component (marker)
    marker = None
    marker_cells = set()
    for cells, colors in components:
        if len(colors) > 1:
            marker = (cells, colors)
            marker_cells = set(cells)
            break
    
    if marker is None:
        return [row[:] for row in grid]
    
    # Parse 2x2 marker pattern
    cells, colors = marker
    min_r = min(r for r,c in cells)
    min_c = min(c for r,c in cells)
    max_r = max(r for r,c in cells)
    max_c = max(c for r,c in cells)
    p_h, p_w = max_r - min_r + 1, max_c - min_c + 1
    
    # Check if it's a valid 2x2 pattern
    if p_h > 2 or p_w > 2:
        return [row[:] for row in grid]
    
    # Read 2x2 pattern
    pattern = [[bg, bg], [bg, bg]]
    for r, c in cells:
        if r - min_r < 2 and c - min_c < 2:
            pattern[r - min_r][c - min_c] = grid[r][c]
    
    # Determine main color (the one that appears in other components)
    other_colors = set()
    for comp_cells, comp_colors in components:
        if set(comp_cells) != marker_cells:
            other_colors |= comp_colors
    main_color = (colors & other_colors).pop() if colors & other_colors else None
    stamp_color = (colors - {main_color} - {bg}).pop() if main_color else None
    
    if main_color is None or stamp_color is None:
        return [row[:] for row in grid]
    
    # Find which position the main color occupies in the 2x2 pattern
    main_pos = None
    for i in range(2):
        for j in range(2):
            if pattern[i][j] == main_color:
                main_pos = (i, j)
    
    # For each single-color component of main_color (excluding marker)
    for cells, colors in components:
        if set(cells) == marker_cells: continue
        if main_color not in colors: continue
        
        # Find bounding box
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        H = max_r - min_r + 1
        W = max_c - min_c + 1
        
        # Determine the rectangle's position in the 2x2 grid
        # main_pos tells us which quadrant the rectangle occupies
        mi, mj = main_pos
        
        # For each quadrant in the 2x2 pattern
        for qi in range(2):
            for qj in range(2):
                if qi == mi and qj == mj:
                    continue  # This is the main block, skip
                
                p_val = pattern[qi][qj]
                if p_val == bg:
                    continue  # BG quadrant, no change
                
                # This quadrant gets stamp_color
                # Calculate position relative to main
                dr = (qi - mi) * H
                dc = (qj - mj) * W
                
                for r in range(min_r + dr, min_r + dr + H):
                    for c in range(min_c + dc, min_c + dc + W):
                        if 0 <= r < R and 0 <= c < C:
                            out[r][c] = stamp_color
    
    return out

