def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    
    # Find special pixel (non-bg)
    special = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                special = (r, c, grid[r][c])
                break
        if special:
            break
    
    output = [row[:] for row in grid]
    
    if special:
        start_row = special[0]
        line_color = special[2]
        # Direction based on column: left edge -> RIGHT, right edge -> LEFT
        direction = 'RIGHT' if special[1] == 0 else 'LEFT'
    else:
        start_row = rows // 2 - 1
        line_color = None
        direction = 'RIGHT'
    
    row = start_row
    while row < rows:
        # Draw horizontal line if we have a line color
        if line_color is not None:
            for c in range(cols):
                output[row][c] = line_color
        
        # Draw corner at row+1
        corner_row = row + 1
        if corner_row < rows:
            if direction == 'RIGHT':
                output[corner_row][cols - 1] = 2
            else:
                output[corner_row][0] = 2
        
        # Flip direction
        direction = 'LEFT' if direction == 'RIGHT' else 'RIGHT'
        row += 2
    
    return output
