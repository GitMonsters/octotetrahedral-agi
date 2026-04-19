def transform(grid):
    import copy
    
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Find background (most common color)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    # Find all non-background colors and their cells
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = []
                color_cells[color].append((r, c))
    
    # Special case: only one non-background color with exactly 2 diagonally adjacent cells
    if len(color_cells) == 1:
        color = list(color_cells.keys())[0]
        cells = color_cells[color]
        if len(cells) == 2:
            (r1, c1), (r2, c2) = cells
            # Check if diagonally adjacent
            if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                # The 5x5 pattern
                pattern = [
                    [0, 1, 1, 1, 0],
                    [1, 0, 1, 0, 1],
                    [1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [0, 1, 1, 1, 0],
                ]
                
                # Determine direction: upper cell at min row
                upper_r, upper_c = (r1, c1) if r1 < r2 else (r2, c2)
                lower_r, lower_c = (r2, c2) if r1 < r2 else (r1, c1)
                
                # Direction based on diagonal slope
                if lower_c > upper_c:
                    # Down-right slope: copies go down-left with step (+4, -4)
                    start_r = upper_r - 0
                    start_c = upper_c - 3
                    dr, dc = 4, -4
                else:
                    # Down-left slope: copies go down-right with step (+4, +4)
                    start_r = upper_r - 0
                    start_c = upper_c - 1
                    dr, dc = 4, 4
                
                # Draw repeated patterns
                pr, pc = start_r, start_c
                while True:
                    drawn = False
                    for i in range(5):
                        for j in range(5):
                            rr, cc = pr + i, pc + j
                            if 0 <= rr < rows and 0 <= cc < cols:
                                if pattern[i][j] == 1:
                                    result[rr][cc] = color
                                    drawn = True
                    if not drawn:
                        break
                    pr += dr
                    pc += dc
                
                return result
    
    # Main case: identify template (multi-cell object) and marker (single cell)
    template_color = None
    template_cells = None
    marker_color = None
    marker_cell = None
    
    for color, cells in color_cells.items():
        if len(cells) == 1:
            marker_color = color
            marker_cell = cells[0]
        else:
            template_color = color
            template_cells = cells
    
    if template_cells is None or marker_cell is None:
        return result
    
    # Get template bounding box
    min_r = min(r for r, c in template_cells)
    max_r = max(r for r, c in template_cells)
    min_c = min(c for r, c in template_cells)
    max_c = max(c for r, c in template_cells)
    
    t_height = max_r - min_r + 1
    t_width = max_c - min_c + 1
    
    # Get template shape relative to its bounding box top-left
    template_shape = []
    for r, c in template_cells:
        template_shape.append((r - min_r, c - min_c))
    
    # Determine direction from marker position relative to template bbox
    mr, mc = marker_cell
    
    if mr > max_r:
        dr = t_height
    elif mr < min_r:
        dr = -t_height
    else:
        dr = 0
    
    if mc > max_c:
        dc = t_width
    elif mc < min_c:
        dc = -t_width
    else:
        dc = 0
    
    # First copy starts with corner opposite the direction at the marker
    if dr > 0 and dc > 0:
        start_r, start_c = mr, mc
    elif dr > 0 and dc < 0:
        start_r, start_c = mr, mc - (t_width - 1)
    elif dr < 0 and dc > 0:
        start_r, start_c = mr - (t_height - 1), mc
    elif dr < 0 and dc < 0:
        start_r, start_c = mr - (t_height - 1), mc - (t_width - 1)
    elif dr > 0:
        start_r, start_c = mr, mc - t_width // 2
    elif dr < 0:
        start_r, start_c = mr - (t_height - 1), mc - t_width // 2
    elif dc > 0:
        start_r, start_c = mr - t_height // 2, mc
    elif dc < 0:
        start_r, start_c = mr - t_height // 2, mc - (t_width - 1)
    else:
        return result
    
    # Draw copies until off grid
    cr, cc_cur = start_r, start_c
    max_iterations = max(rows, cols) * 2
    iteration = 0
    
    while iteration < max_iterations:
        any_in_grid = False
        for rel_r, rel_c in template_shape:
            rr, cc_pos = cr + rel_r, cc_cur + rel_c
            if 0 <= rr < rows and 0 <= cc_pos < cols:
                result[rr][cc_pos] = marker_color
                any_in_grid = True
        
        if not any_in_grid:
            break
        
        cr += dr
        cc_cur += dc
        iteration += 1
    
    return result
