def solve(grid):
    """
    Transform the grid by creating horizontal reflection pattern with color 3.
    
    Based on analysis:
    - Find the main pattern (non-background cells)
    - Find pattern bounding box 
    - For background cells in lower half of pattern bbox, if the horizontally 
      reflected position contains a pattern cell, change to color 3
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
    
    # Calculate horizontal center for reflection
    center_c = (min_c + max_c) / 2.0
    
    # Key insight: Only consider lower portion of pattern for transformation
    # Based on observation that changes occur in bottom area
    pattern_height = max_r - min_r + 1
    middle_r = min_r + pattern_height // 2
    
    # For each background cell in the pattern bounding box,
    # particularly in the lower half
    for r in range(middle_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:  # This is a background cell
                # Find horizontally reflected position
                reflected_c = int(round(2 * center_c - c))
                
                # If reflected position is within bounds and has pattern
                if (min_c <= reflected_c <= max_c and 
                    (r, reflected_c) in pattern_cells):
                    result[r][c] = 3
    
    return result