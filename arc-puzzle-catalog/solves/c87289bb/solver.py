def solve(grid):
    """
    Transform rule:
    1. Find the row with 2s (the "object")
    2. Find the bounding box columns of those 2s (min_col and max_col)
    3. Find all columns that have 8s in the top rows (8-columns)
    4. Create a rectangular box by:
       - Top border (one row above 2s): fill certain columns with 8s
       - Bottom border (2s row and below): fill certain columns with 8s
    """
    import copy
    
    grid = [list(row) for row in grid]
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    # Find the row containing 2s
    obj_row = -1
    obj_cols = []
    for r in range(height):
        for c in range(width):
            if grid[r][c] == 2:
                if obj_row == -1:
                    obj_row = r
                obj_cols.append(c)
    
    if obj_row == -1:
        return grid
    
    min_col_2 = min(obj_cols)
    max_col_2 = max(obj_cols)
    above_row = obj_row - 1
    
    if above_row < 0:
        return grid
    
    # Find all 8-columns (columns with 8 in any row above obj_row)
    eight_cols = set()
    for r in range(obj_row):
        for c in range(width):
            if grid[r][c] == 8:
                eight_cols.add(c)
    
    # Outer 8-columns (not in the 2s range)
    outer_eights = {c for c in eight_cols if c < min_col_2 or c > max_col_2}
    
    # Inner 8-columns (within 2s range)
    inner_eights = {c for c in eight_cols if min_col_2 <= c <= max_col_2}
    
    # Build the top border (row above 2s)
    # Include: all columns with 8 that exist above + fill to max_col_2+1 + special rules
    new_above = eight_cols.copy()
    new_above.add(max_col_2 + 1)  # Always add right edge
    
    # For the range between left and right bounds, add all columns in [min_col+1, max_col+1]
    # But based on examples, we need a more complex rule...
    # For now, use: outer eights + inner eights + range from min_col_2+1 to max_col_2+1
    eight_cols_sorted = sorted(eight_cols)
    for c in range(min_col_2 + 1, max_col_2 + 2):
        new_above.add(c)
    
    # For the left side before min_col_2:
    # Include all 8-columns that are < min_col_2
    for c in [c for c in eight_cols if c < min_col_2]:
        new_above.add(c)
    
    # Fill the above row
    for c in range(width):
        if grid[above_row][c] == 0 and c in new_above:
            grid[above_row][c] = 8
    
    # Build the bottom border (2s row and below)
    # Based on analysis: new_below = new_above - inner_eights - {max_col_2}
    new_below = new_above - inner_eights - {max_col_2}
    
    # Fill from the row with 2s down to the bottom
    for r in range(obj_row, height):
        for c in range(width):
            if grid[r][c] == 0 and c in new_below:
                grid[r][c] = 8
    
    return grid

