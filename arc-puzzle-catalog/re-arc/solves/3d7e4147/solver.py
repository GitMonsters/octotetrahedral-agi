from collections import Counter

def transform(grid):
    """
    Pattern: Find a solid rectangle (largest contiguous block of uniform non-background color).
    For each row/column aligned with that rectangle, find the farthest 
    non-background color in each direction and extend it back to the rectangle edge.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    all_vals = [v for row in grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    def find_largest_solid_rectangle():
        """Find the largest contiguous rectangular block of a single non-bg color"""
        best_area = 0
        best_rect = None
        best_color = None
        
        # For each possible top-left corner
        for r1 in range(rows):
            for c1 in range(cols):
                color = grid[r1][c1]
                if color == bg:
                    continue
                
                # Try to extend as far right as possible
                max_c2 = cols - 1
                for r2 in range(r1, rows):
                    # Check how far right we can go on this row
                    c2 = c1
                    while c2 <= max_c2 and grid[r2][c2] == color:
                        c2 += 1
                    c2 -= 1  # Back up one since we went one past
                    
                    if c2 < c1:
                        # Can't even include c1 in this row
                        break
                    
                    max_c2 = c2  # Shrink max_c2 for future rows
                    
                    # Check area of rectangle from (r1,c1) to (r2,c2)
                    area = (r2 - r1 + 1) * (max_c2 - c1 + 1)
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, r2, c1, max_c2)
                        best_color = color
        
        return best_rect, best_color
    
    rect, rect_color = find_largest_solid_rectangle()
    if rect is None:
        return grid
    
    r1, r2, c1, c2 = rect
    
    # Create output grid
    out = [row[:] for row in grid]
    
    # For each row in the rectangle range, extend left and right
    for r in range(r1, r2 + 1):
        # Extend LEFT: find leftmost non-bg color to the left of rectangle
        leftmost_col = None
        leftmost_color = None
        for c in range(c1 - 1, -1, -1):
            if grid[r][c] != bg:
                leftmost_col = c
                leftmost_color = grid[r][c]
        if leftmost_col is not None:
            for c in range(leftmost_col, c1):
                out[r][c] = leftmost_color
        
        # Extend RIGHT: find rightmost non-bg color to the right of rectangle
        rightmost_col = None
        rightmost_color = None
        for c in range(c2 + 1, cols):
            if grid[r][c] != bg:
                rightmost_col = c
                rightmost_color = grid[r][c]
        if rightmost_col is not None:
            for c in range(c2 + 1, rightmost_col + 1):
                out[r][c] = rightmost_color
    
    # For each column in the rectangle range, extend up and down
    for c in range(c1, c2 + 1):
        # Extend UP: find topmost non-bg color above rectangle
        topmost_row = None
        topmost_color = None
        for r in range(r1 - 1, -1, -1):
            if grid[r][c] != bg:
                topmost_row = r
                topmost_color = grid[r][c]
        if topmost_row is not None:
            for r in range(topmost_row, r1):
                out[r][c] = topmost_color
        
        # Extend DOWN: find bottommost non-bg color below rectangle
        bottommost_row = None
        bottommost_color = None
        for r in range(r2 + 1, rows):
            if grid[r][c] != bg:
                bottommost_row = r
                bottommost_color = grid[r][c]
        if bottommost_row is not None:
            for r in range(r2 + 1, bottommost_row + 1):
                out[r][c] = bottommost_color
    
    return out
