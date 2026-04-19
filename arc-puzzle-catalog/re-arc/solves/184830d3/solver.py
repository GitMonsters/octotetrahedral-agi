def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common color)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_count[color] = color_count.get(color, 0) + 1
    
    background = max(color_count.keys(), key=lambda x: color_count[x])
    
    # Find all non-background colored cells
    colored_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                colored_cells.append((r, c, grid[r][c]))
    
    # Create output grid (copy of input)
    output = [row[:] for row in grid]
    
    # For each colored cell, draw a vertical line through the entire column
    for r, c, color in colored_cells:
        for row_idx in range(rows):
            output[row_idx][c] = color
    
    # Now handle the pattern where we need to extend to the left edge
    # Looking at examples more carefully:
    # Example 1: Two points at (16,6) color 2 and (16,12) color 4
    #   Output has lines at columns 0, 6, 12 with colors 4, 2, 4
    # Example 2: Two points at (0,12) color 2 and (15,6) color 2
    #   Output has lines at columns 0, 6, 12 all color 2
    
    # The pattern seems to be: find the spacing between the colored columns,
    # and repeat that pattern to the left (to column 0)
    
    if len(colored_cells) >= 2:
        # Get unique columns with their colors
        col_to_color = {}
        for r, c, color in colored_cells:
            col_to_color[c] = color
        
        cols_sorted = sorted(col_to_color.keys())
        
        if len(cols_sorted) >= 2:
            # Calculate the spacing
            spacing = cols_sorted[1] - cols_sorted[0]
            
            # Extend pattern to the left
            leftmost_col = cols_sorted[0]
            leftmost_color = col_to_color[leftmost_col]
            
            # The rightmost point's color is used for the extended lines
            rightmost_col = cols_sorted[-1]
            rightmost_color = col_to_color[rightmost_col]
            
            # Extend leftward from the leftmost column
            current_col = leftmost_col - spacing
            while current_col >= 0:
                for row_idx in range(rows):
                    output[row_idx][current_col] = rightmost_color
                current_col -= spacing
    
    elif len(colored_cells) == 1:
        # Single point - just draw vertical line (already done above)
        pass
    
    return output