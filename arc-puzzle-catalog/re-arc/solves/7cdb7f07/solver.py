def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find connected components of non-bg cells
    visited = [[False]*cols for _ in range(rows)]
    components = []
    
    def flood_fill(r, c, val):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols: continue
            if visited[cr][cc] or grid[cr][cc] != val: continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                comp = flood_fill(r, c, grid[r][c])
                if comp:
                    components.append((grid[r][c], comp))
    
    walls = []
    markers = []
    for val, comp in components:
        if len(comp) >= 2:
            walls.append((val, comp))
        else:
            markers.append((comp[0][0], comp[0][1], val))
    
    output = [row[:] for row in grid]
    marker_color = markers[0][2] if markers else 0
    
    if len(walls) == 2:
        # Two-wall case
        wall1_val, wall1_cells = walls[0]
        wall2_val, wall2_cells = walls[1]
        
        # Determine wall orientation and properties
        def wall_props(cells):
            rmin = min(r for r,c in cells)
            rmax = max(r for r,c in cells)
            cmin = min(c for r,c in cells)
            cmax = max(c for r,c in cells)
            if rmin == rmax:
                return 'horizontal', rmin, cmin, cmax
            elif cmin == cmax:
                return 'vertical', cmin, rmin, rmax
            return 'other', 0, 0, 0
        
        w1_orient, w1_pos, w1_start, w1_end = wall_props(wall1_cells)
        w2_orient, w2_pos, w2_start, w2_end = wall_props(wall2_cells)
        
        # Determine near wall (the one whose span overlaps with marker positions)
        marker_positions = [(r, c) for r, c, v in markers]
        
        def markers_near_wall(orient, pos, start, end):
            if orient == 'horizontal':
                return all(start <= c <= end for r, c in marker_positions)
            else:
                return all(start <= r <= end for r, c in marker_positions)
        
        if markers_near_wall(w1_orient, w1_pos, w1_start, w1_end):
            near_orient, near_pos, near_start, near_end = w1_orient, w1_pos, w1_start, w1_end
            far_orient, far_pos, far_start, far_end = w2_orient, w2_pos, w2_start, w2_end
        else:
            near_orient, near_pos, near_start, near_end = w2_orient, w2_pos, w2_start, w2_end
            far_orient, far_pos, far_start, far_end = w1_orient, w1_pos, w1_start, w1_end
        
        for mr, mc, mv in markers:
            output[mr][mc] = 1  # Replace marker with 1
            
            if near_orient == 'horizontal':
                # Horizontal wall: lines are vertical
                offset = mc - near_start
                
                # Near wall line: fill gap between wall and marker
                if near_pos < mr:
                    for r in range(near_pos + 1, mr):
                        output[r][mc] = mv
                else:
                    for r in range(mr + 1, near_pos):
                        output[r][mc] = mv
                
                # Far wall line: at corresponding offset, full span to opposite edge
                far_col = far_start + offset
                if far_pos < near_pos:
                    # Far wall is above near wall; lines go up from far wall
                    for r in range(0, far_pos):
                        output[r][far_col] = mv
                else:
                    # Far wall is below near wall; lines go down from far wall
                    for r in range(far_pos + 1, rows):
                        output[r][far_col] = mv
                    # Also go up to edge
                    for r in range(0, far_pos):
                        output[r][far_col] = mv
            
            else:  # vertical wall
                offset = mr - near_start
                
                # Near wall line: fill gap between wall and marker
                if near_pos < mc:
                    for c in range(near_pos + 1, mc):
                        output[mr][c] = mv
                else:
                    for c in range(mc + 1, near_pos):
                        output[mr][c] = mv
                
                # Far wall line: at corresponding offset, full span to opposite edge
                far_row = far_start + offset
                if far_pos < near_pos:
                    for c in range(0, far_pos):
                        output[far_row][c] = mv
                else:
                    for c in range(far_pos + 1, cols):
                        output[far_row][c] = mv
                    for c in range(0, far_pos):
                        output[far_row][c] = mv
    
    elif len(walls) == 0:
        # No-wall case (like Train 0)
        # Lines go from each marker downward to rows-2
        # Border at cols max_col+2 to cols-1
        max_col = max(c for r, c, v in markers)
        min_row = min(r for r, c, v in markers)
        
        for mr, mc, mv in markers:
            output[mr][mc] = 1
            for r in range(mr + 1, rows - 1):
                output[r][mc] = mv
        
        # Border
        border_start = max_col + 2
        for r in range(min_row, rows):
            for c in range(border_start, cols):
                output[r][c] = marker_color
    
    return output
