import numpy as np

def transform(grid):
    """
    Transform an ARC puzzle grid by replacing rectangle edge cells.
    
    The algorithm:
    1. Finds the main rectangle (non-7 color with >= 99% fill of bounding box)
    2. For each edge cell, determines if it should be transformed
    3. Uses markers from grid borders (row 0, row -1, col 0, col -1) to determine replacement values
    """
    grid = np.array(grid, dtype=int)
    output = grid.copy()
    
    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg = unique[np.argmax(counts)]
    
    # Find the rectangle color (non-7, non-bg color with fill ratio >= 0.99)
    rect_color = None
    rect_bbox = None
    
    for color in np.unique(grid):
        if color == 7 or color == bg:
            continue
        
        positions = np.where(grid == color)
        if len(positions[0]) == 0:
            continue
        
        min_r, max_r = positions[0].min(), positions[0].max()
        min_c, max_c = positions[1].min(), positions[1].max()
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        count = len(positions[0])
        fill_ratio = count / area
        
        if fill_ratio >= 0.99:
            rect_color = color
            rect_bbox = (min_r, max_r, min_c, max_c)
            break
    
    if rect_color is None:
        return output
    
    min_row, max_row, min_col, max_col = rect_bbox
    
    # Transform each cell on the rectangle boundary
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if output[row, col] != rect_color:
                continue
            
            is_top = row == min_row
            is_bottom = row == max_row
            is_left = col == min_col
            is_right = col == max_col
            
            # Skip interior cells
            if not (is_top or is_bottom or is_left or is_right):
                continue
            
            # Get border markers
            left_marker = output[row, 0]
            right_marker = output[row, -1]
            top_marker = output[0, col]
            bottom_marker = output[-1, col]
            
            replacement = None
            
            # For corner cells, prefer special markers
            if (is_top or is_bottom) and (is_left or is_right):
                # This is a corner
                # Priority depends on which corner
                if is_bottom and is_right:
                    # BR: prefer right, then bottom, then others
                    if right_marker != 7 and right_marker != bg:
                        replacement = right_marker
                    elif bottom_marker != 7 and bottom_marker != bg:
                        replacement = bottom_marker
                    elif right_marker != 7:
                        replacement = right_marker
                    elif bottom_marker != 7:
                        replacement = bottom_marker
                else:
                    # For other corners: prefer any special marker
                    if top_marker != 7 and top_marker != bg:
                        replacement = top_marker
                    elif bottom_marker != 7 and bottom_marker != bg:
                        replacement = bottom_marker
                    elif left_marker != 7 and left_marker != bg:
                        replacement = left_marker
                    elif right_marker != 7 and right_marker != bg:
                        replacement = right_marker
                    elif left_marker != 7:
                        replacement = left_marker
                    elif right_marker != 7:
                        replacement = right_marker
                    elif top_marker != 7:
                        replacement = top_marker
                    elif bottom_marker != 7:
                        replacement = bottom_marker
            
            # For non-corner edge cells
            elif is_top:
                if top_marker != 7 and top_marker != bg:
                    replacement = top_marker
            
            elif is_bottom:
                if bottom_marker != 7 and bottom_marker != bg:
                    replacement = bottom_marker
            
            elif is_left:
                if left_marker != 7 and left_marker != bg:
                    replacement = left_marker
            
            elif is_right:
                if right_marker != 7 and right_marker != bg:
                    replacement = right_marker
            
            if replacement is not None:
                output[row, col] = replacement
    
    return output.tolist()
