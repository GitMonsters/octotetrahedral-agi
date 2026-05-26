def transform(grid):
    """
    NEW HYPOTHESIS: The transformation creates reflections of non-background cells,
    but it might not be simple global reflections. Let me try creating all possible 
    reflections: across center, quarter lines, etc.
    """
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]  # Background color
    
    # Copy the input grid
    result = [row[:] for row in grid]
    
    # Collect all non-background cells
    cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                cells.append((r, c, grid[r][c]))
    
    # For each cell, try multiple types of reflections
    for r, c, val in cells:
        # Simple global reflections across grid center
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        
        # Horizontal reflection (across vertical center)
        hc = cols - 1 - c
        if 0 <= r < rows and 0 <= hc < cols and result[r][hc] == bg:
            result[r][hc] = val
        
        # Vertical reflection (across horizontal center)  
        vr = rows - 1 - r
        if 0 <= vr < rows and 0 <= c < cols and result[vr][c] == bg:
            result[vr][c] = val
            
        # Point reflection (180 degrees around center)
        br = rows - 1 - r
        bc = cols - 1 - c
        if 0 <= br < rows and 0 <= bc < cols and result[br][bc] == bg:
            result[br][bc] = val
    
    # Additional step: maybe it also creates intermediate reflections
    # Let's try reflecting the already created reflections  
    new_cells = []
    for r in range(rows):
        for c in range(cols):
            if result[r][c] != bg and grid[r][c] == bg:  # New cells we just added
                new_cells.append((r, c, result[r][c]))
    
    # Create reflections of the new cells too
    for r, c, val in new_cells:
        # Horizontal reflection
        hc = cols - 1 - c
        if 0 <= r < rows and 0 <= hc < cols and result[r][hc] == bg:
            result[r][hc] = val
        
        # Vertical reflection
        vr = rows - 1 - r
        if 0 <= vr < rows and 0 <= c < cols and result[vr][c] == bg:
            result[vr][c] = val
            
        # Point reflection
        br = rows - 1 - r
        bc = cols - 1 - c
        if 0 <= br < rows and 0 <= bc < cols and result[br][bc] == bg:
            result[br][bc] = val
    
    return result

solve = transform