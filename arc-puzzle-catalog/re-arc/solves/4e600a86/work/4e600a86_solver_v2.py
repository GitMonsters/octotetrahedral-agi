def solve(grid):
    """
    Create horizontal reflection of the entire pattern.
    
    The transformation appears to:
    1. Find the pattern (non-background cells)
    2. Create a horizontal mirror/reflection of the entire pattern 
    3. Fill background cells with color 3 where the reflected pattern would be
    4. Only fill if the location is background and within reasonable bounds
    """
    import copy
    
    if not grid or not grid[0]:
        return grid
    
    result = copy.deepcopy(grid)
    h, w = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    colors = {}
    for row in grid:
        for cell in row:
            colors[cell] = colors.get(cell, 0) + 1
    bg_color = max(colors, key=colors.get)
    
    # If background is already 3, no transformation needed
    if bg_color == 3:
        return result
    
    # Find pattern cells (non-background)
    pattern_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg_color:
                pattern_cells.add((r, c))
    
    if not pattern_cells:
        return result
    
    # Find pattern bounding box
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)  
    max_c = max(c for r, c in pattern_cells)
    
    # Calculate center for horizontal reflection
    center_c = (min_c + max_c) / 2.0
    
    # For each pattern cell, compute its reflection
    reflected_positions = set()
    for r, c in pattern_cells:
        reflected_c = int(round(2 * center_c - c))
        reflected_positions.add((r, reflected_c))
    
    # Fill background cells with 3 where reflected pattern would be
    # But only within the original pattern's row range and reasonable column range
    for r, c in reflected_positions:
        if (0 <= r < h and 0 <= c < w and 
            grid[r][c] == bg_color and
            min_r <= r <= max_r):  # Stay within pattern's vertical range
            result[r][c] = 3
    
    return result