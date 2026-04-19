def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a copy of the grid
    result = [row[:] for row in grid]
    
    # Find the background color (most common color in the interior)
    bg_color = grid[rows // 2][cols // 2]
    
    # Find rows and columns that have non-background markers on both ends
    # For each such row, we fill it with the left marker value on the left half
    # and right marker value on the right half, with a separator (color 8) in the middle
    
    # First, identify rows that have non-bg values on edges
    marked_rows = []
    for r in range(rows):
        left_val = grid[r][0]
        right_val = grid[r][cols - 1]
        if left_val != bg_color or right_val != bg_color:
            marked_rows.append(r)
    
    # Identify columns that have non-bg values on edges
    marked_cols = []
    for c in range(cols):
        top_val = grid[0][c]
        bottom_val = grid[rows - 1][c]
        if top_val != bg_color or bottom_val != bg_color:
            marked_cols.append(c)
    
    # Determine the center column (where the separator goes)
    # Looking at example 1: center is at column 7 (middle of 15 cols)
    # Looking at example 2: center is at column 11 (middle of 23 cols)
    center_col = cols // 2
    center_row = rows // 2
    
    # For each marked row, fill it
    for r in marked_rows:
        left_val = grid[r][0]
        right_val = grid[r][cols - 1]
        
        # If both ends are background, skip
        if left_val == bg_color and right_val == bg_color:
            continue
        
        # Fill left half with left_val (if not bg) up to center
        # Fill right half with right_val (if not bg) from center+1
        # Put 8 at center
        
        if left_val != bg_color and right_val != bg_color:
            # Both sides have markers
            for c in range(cols):
                if c < center_col:
                    result[r][c] = left_val
                elif c == center_col:
                    result[r][c] = 8
                else:
                    result[r][c] = right_val
        elif left_val != bg_color:
            # Only left side has marker
            for c in range(cols):
                if c < center_col:
                    result[r][c] = left_val
                elif c == center_col:
                    result[r][c] = 8
                else:
                    result[r][c] = bg_color
        elif right_val != bg_color:
            # Only right side has marker
            for c in range(cols):
                if c < center_col:
                    result[r][c] = bg_color
                elif c == center_col:
                    result[r][c] = 8
                else:
                    result[r][c] = right_val
    
    # Similar for columns if needed (example 2 seems to be row-based only)
    # Actually looking more carefully at example 2, it seems like the same row-based logic
    
    return result