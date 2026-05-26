def solve(grid):
    """
    Vertical reflection with careful center calculation and no arbitrary constraints.
    """
    import copy
    
    result = copy.deepcopy(grid)
    
    if not grid or not grid[0]:
        return result
    
    h, w = len(grid), len(grid[0])
    
    # Find background color (most frequent)  
    colors = {}
    for row in grid:
        for cell in row:
            colors[cell] = colors.get(cell, 0) + 1
    bg_color = max(colors, key=colors.get)
    
    # If background is already 3, no change needed
    if bg_color == 3:
        return result
    
    # Find all non-background cells (the pattern)
    pattern_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg_color:
                pattern_cells.add((r, c))
    
    if not pattern_cells:
        return result
    
    # Find bounding box of the pattern
    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)
    max_c = max(c for r, c in pattern_cells)
    
    # Use the original approach but with exact center calculation
    pattern_height = max_r - min_r + 1
    center_r = min_r + (pattern_height - 1) / 2.0
    
    # Apply vertical reflection
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:
                # Calculate reflected row
                dist_from_center = r - center_r
                reflected_r = center_r - dist_from_center
                reflected_r_int = int(round(reflected_r))
                
                # Check if the reflected position contains a pattern cell
                if (min_r <= reflected_r_int <= max_r and 
                    (reflected_r_int, c) in pattern_cells):
                    result[r][c] = 3
    
    return result