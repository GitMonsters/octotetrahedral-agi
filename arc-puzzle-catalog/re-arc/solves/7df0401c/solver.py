def transform(grid):
    """
    Find L-shaped pieces and combine them into a rectangle outline.
    Each L-shape represents a corner of the output rectangle.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all colored cells grouped by color
    all_colors = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val not in all_colors:
                all_colors[val] = 0
            all_colors[val] += 1
    
    # Identify background color (most common color, could be 0 or another)
    background = max(all_colors.keys(), key=lambda c: all_colors[c])
    
    # Group non-background, non-zero cells by color
    # (0 can be holes/gaps in the background, not actual L-shapes)
    colors = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != background and val != 0:
                if val not in colors:
                    colors[val] = []
                colors[val].append((r, c))
    
    def classify_l_shape(cells):
        """Classify L-shape corner type and get arm lengths"""
        r_coords = [c[0] for c in cells]
        c_coords = [c[1] for c in cells]
        min_r, max_r = min(r_coords), max(r_coords)
        min_c, max_c = min(c_coords), max(c_coords)
        
        # Count cells in each row/col
        row_counts = {}
        col_counts = {}
        for r, c in cells:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        
        # The horizontal arm is the row with most cells
        max_row_count = max(row_counts.values())
        max_col_count = max(col_counts.values())
        
        horiz_row = [r for r, cnt in row_counts.items() if cnt == max_row_count][0]
        vert_col = [c for c, cnt in col_counts.items() if cnt == max_col_count][0]
        
        # Determine corner type based on where arms originate
        if horiz_row == min_r and vert_col == min_c:
            corner_type = 'TL'
        elif horiz_row == min_r and vert_col == max_c:
            corner_type = 'TR'
        elif horiz_row == max_r and vert_col == min_c:
            corner_type = 'BL'
        else:
            corner_type = 'BR'
        
        return corner_type, max_row_count, max_col_count
    
    # Classify all L-shapes
    shapes = {}
    for color, cells in colors.items():
        corner_type, h_arm, v_arm = classify_l_shape(cells)
        shapes[corner_type] = (color, h_arm, v_arm)
    
    # Determine output dimensions
    # Get h_arm and v_arm for each side
    left_h = 0   # h_arm from TL or BL
    right_h = 0  # h_arm from TR or BR
    top_v = 0    # v_arm from TL or TR
    bottom_v = 0 # v_arm from BL or BR
    
    has_top = 'TL' in shapes or 'TR' in shapes
    has_bottom = 'BL' in shapes or 'BR' in shapes
    has_left = 'TL' in shapes or 'BL' in shapes
    has_right = 'TR' in shapes or 'BR' in shapes
    
    for corner_type, (color, h_arm, v_arm) in shapes.items():
        if corner_type == 'TL':
            left_h = max(left_h, h_arm)
            top_v = max(top_v, v_arm)
        elif corner_type == 'TR':
            right_h = max(right_h, h_arm)
            top_v = max(top_v, v_arm)
        elif corner_type == 'BL':
            left_h = max(left_h, h_arm)
            bottom_v = max(bottom_v, v_arm)
        elif corner_type == 'BR':
            right_h = max(right_h, h_arm)
            bottom_v = max(bottom_v, v_arm)
    
    # Width = sum of left and right h_arms
    if has_left and has_right:
        out_width = left_h + right_h
    else:
        out_width = max(left_h, right_h)
    
    # Height calculation depends on whether we have top and bottom corners
    if has_top and has_bottom:
        # Overlap of 1 row where top and bottom meet
        out_height = top_v + bottom_v - 1
    else:
        # Only one side - height is the max of h_arms (forms a square-ish region)
        all_h_arms = [shapes[ct][1] for ct in shapes]
        out_height = max(all_h_arms)
    
    # Create output grid
    output = [[0 for _ in range(out_width)] for _ in range(out_height)]
    
    # Draw each L-shape in its corner position
    for corner_type, (color, h_arm, v_arm) in shapes.items():
        if corner_type == 'TL':
            # Draw horizontal arm at top-left
            for c in range(h_arm):
                output[0][c] = color
            # Draw vertical arm down from top-left
            for r in range(v_arm):
                output[r][0] = color
        elif corner_type == 'TR':
            # Draw horizontal arm at top-right
            for c in range(h_arm):
                output[0][out_width - 1 - c] = color
            # Draw vertical arm down from top-right
            for r in range(v_arm):
                output[r][out_width - 1] = color
        elif corner_type == 'BL':
            # Draw horizontal arm at bottom-left
            for c in range(h_arm):
                output[out_height - 1][c] = color
            # Draw vertical arm up from bottom-left
            for r in range(v_arm):
                output[out_height - 1 - r][0] = color
        elif corner_type == 'BR':
            # Draw horizontal arm at bottom-right
            for c in range(h_arm):
                output[out_height - 1][out_width - 1 - c] = color
            # Draw vertical arm up from bottom-right
            for r in range(v_arm):
                output[out_height - 1 - r][out_width - 1] = color
    
    return output
