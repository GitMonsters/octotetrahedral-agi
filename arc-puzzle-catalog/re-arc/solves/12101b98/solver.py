def transform(grid):
    import copy
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg_color = Counter(flat).most_common(1)[0][0]
    
    # Find connected components
    visited = [[False]*cols for _ in range(rows)]
    
    def flood_fill(r, c, color):
        stack = [(r, c)]
        cells = set()
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc]:
                continue
            if grid[cr][cc] != color:
                continue
            visited[cr][cc] = True
            cells.add((cr, cc))
            stack.extend([(cr-1,cc), (cr+1,cc), (cr,cc-1), (cr,cc+1)])
        return cells
    
    shapes = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg_color:
                color = grid[r][c]
                cells = flood_fill(r, c, color)
                if cells:
                    shapes.append((color, cells))
    
    # Find markers: single pixels inside larger shapes of different color
    markers = []  # (marker_r, marker_c, marker_color, shape_color, shape_cells)
    
    for i, (color, cells) in enumerate(shapes):
        if len(cells) > 3:  # A real shape
            # Find any single pixel of different color that is surrounded/adjacent to this shape
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            
            # Look for single pixels inside bounding box
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if (r, c) not in cells and grid[r][c] != bg_color:
                        # Check if surrounded by shape cells
                        adj_count = 0
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in cells:
                                adj_count += 1
                        if adj_count >= 2:  # At least 2 adjacent shape cells
                            markers.append((r, c, grid[r][c], color, cells))
    
    # For each marker, project the row/column at marker position
    for mr, mc, marker_color, shape_color, shape_cells in markers:
        min_r = min(r for r, c in shape_cells)
        max_r = max(r for r, c in shape_cells)
        min_c = min(c for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        
        # Get the row of the shape at marker row
        row_cells_at_mr = [(r, c) for r, c in shape_cells if r == mr]
        col_cells_at_mc = [(r, c) for r, c in shape_cells if c == mc]
        
        # Determine projection direction
        # If marker is in top half of shape, project up; bottom half, project down
        # If marker is in left half of shape, project left; right half, project right
        
        mid_r = (min_r + max_r) / 2
        mid_c = (min_c + max_c) / 2
        
        # Project in the direction away from center
        if row_cells_at_mr:
            c_vals = [c for r, c in row_cells_at_mr]
            c_min, c_max = min(c_vals), max(c_vals)
            
            if mr <= mid_r:  # top half - project up
                for nr in range(mr - 1, -1, -1):
                    for cc in range(c_min, c_max + 1):
                        if output[nr][cc] == bg_color:
                            output[nr][cc] = shape_color
                    if output[nr][mc] == bg_color or output[nr][mc] == shape_color:
                        output[nr][mc] = marker_color
            
            if mr >= mid_r:  # bottom half - project down
                for nr in range(mr + 1, rows):
                    for cc in range(c_min, c_max + 1):
                        if output[nr][cc] == bg_color:
                            output[nr][cc] = shape_color
                    if output[nr][mc] == bg_color or output[nr][mc] == shape_color:
                        output[nr][mc] = marker_color
        
        if col_cells_at_mc:
            r_vals = [r for r, c in col_cells_at_mc]
            r_min, r_max = min(r_vals), max(r_vals)
            
            if mc <= mid_c:  # left half - project left
                for nc in range(mc - 1, -1, -1):
                    for rr in range(r_min, r_max + 1):
                        if output[rr][nc] == bg_color:
                            output[rr][nc] = shape_color
                    if output[mr][nc] == bg_color or output[mr][nc] == shape_color:
                        output[mr][nc] = marker_color
            
            if mc >= mid_c:  # right half - project right
                for nc in range(mc + 1, cols):
                    for rr in range(r_min, r_max + 1):
                        if output[rr][nc] == bg_color:
                            output[rr][nc] = shape_color
                    if output[mr][nc] == bg_color or output[mr][nc] == shape_color:
                        output[mr][nc] = marker_color
    
    return output
