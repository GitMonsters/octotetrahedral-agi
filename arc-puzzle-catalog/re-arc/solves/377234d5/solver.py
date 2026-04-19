def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Create output grid filled with background (1)
    output = [[1 for _ in range(cols)] for _ in range(rows)]
    
    # Find the rectangular region (non-1 and non-6 colored block)
    # Also find all the 6s (markers)
    rect_color = None
    rect_top, rect_left, rect_bottom, rect_right = None, None, None, None
    
    markers = []  # positions of 6s
    
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val == 6:
                markers.append((r, c))
            elif val != 1:
                # Part of the rectangle
                if rect_color is None:
                    rect_color = val
                    rect_top = rect_bottom = r
                    rect_left = rect_right = c
                else:
                    rect_top = min(rect_top, r)
                    rect_bottom = max(rect_bottom, r)
                    rect_left = min(rect_left, c)
                    rect_right = max(rect_right, c)
    
    if rect_color is None:
        return output
    
    # Copy the rectangle to output
    for r in range(rect_top, rect_bottom + 1):
        for c in range(rect_left, rect_right + 1):
            output[r][c] = rect_color
    
    # Now we need to "fold" or "reflect" the markers toward the rectangle
    # Each marker gets moved to be adjacent to the rectangle
    # The transformation moves markers to positions just outside the rectangle boundary
    
    # For each marker, determine which side of the rectangle it would project to
    # and place it at the corresponding position adjacent to the rectangle
    
    for (mr, mc) in markers:
        # Determine the row position relative to rectangle
        if mr < rect_top:
            # Above the rectangle - project to row just above rectangle
            new_r = rect_top - 1
        elif mr > rect_bottom:
            # Below the rectangle - project to row just below rectangle
            new_r = rect_bottom + 1
        else:
            # Within the rectangle's row span - keep same row
            new_r = mr
        
        # Determine the column position relative to rectangle
        if mc < rect_left:
            # Left of the rectangle - project to column just left of rectangle
            new_c = rect_left - 1
        elif mc > rect_right:
            # Right of the rectangle - project to column just right of rectangle
            new_c = rect_right + 1
        else:
            # Within the rectangle's column span - keep same column
            new_c = mc
        
        # Place the marker at the new position
        if 0 <= new_r < rows and 0 <= new_c < cols:
            output[new_r][new_c] = 6
    
    return output