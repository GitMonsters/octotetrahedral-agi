def solve(grid):
    """
    Complete horizontal symmetry by reflecting pattern across its center.
    
    Looking at the visual patterns, it appears the transformation:
    1. Finds the pattern bounding box
    2. For each pattern cell, computes its horizontal reflection across the bbox center
    3. If the reflected position is background (within the bbox), change it to color 3
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
    
    # Calculate center for horizontal reflection within bounding box
    center_c = (min_c + max_c) / 2.0
    
    # For each pattern cell, compute its reflection and fill if it's background
    for r, c in pattern_cells:
        # Reflect across the bounding box center
        reflected_c = int(round(2 * center_c - c))
        
        # If reflected position is within bounds and is background
        if (min_c <= reflected_c <= max_c and 
            0 <= r < h and 0 <= reflected_c < w and
            grid[r][reflected_c] == bg_color):
            result[r][reflected_c] = 3
    
    return result