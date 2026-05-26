def solve(grid):
    """
    Apply vertical reflection with additional constraints to match exactly.
    
    Based on analysis showing 84% recall, need to be more selective.
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
    
    # Calculate the vertical center
    center_r = (min_r + max_r) / 2.0
    
    # Additional constraint: only fill positions that are "gaps" in existing structure
    # Look for background cells that would create better symmetry
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == bg_color:
                # Find the vertically reflected position
                reflected_r = int(round(2 * center_r - r))
                
                # Check if reflection is valid and creates meaningful symmetry
                if (min_r <= reflected_r <= max_r and 
                    (reflected_r, c) in pattern_cells):
                    
                    # Additional filtering: maybe only fill if it's in the "lower" half
                    # or if it helps complete a structural element
                    if r > center_r:  # Only fill in lower half of pattern
                        result[r][c] = 3
    
    return result