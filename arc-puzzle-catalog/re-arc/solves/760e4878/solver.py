def transform(grid):
    """
    Two patterns:
    1. If 4 corner markers exist: extract region inside, swap fg->marker color, keep bg
       - Background is the color that appears in the entire grid (not just region)
    2. Otherwise: find blob, return bounding box filled with background
       - If boundary columns have isolated pixels (count=1), add 1 to width
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Count colors and positions
    color_counts = {}
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            color_counts[v] = color_counts.get(v, 0) + 1
            if v not in color_positions:
                color_positions[v] = []
            color_positions[v].append((r, c))
    
    # Global background is most common color overall
    sorted_global = sorted(color_counts.items(), key=lambda x: -x[1])
    global_bg = sorted_global[0][0]
    
    # Look for exactly 4 pixels forming rectangle corners (marker color)
    marker_color = None
    corners = None
    for color, positions in color_positions.items():
        if len(positions) == 4:
            rs = sorted(set(p[0] for p in positions))
            cs = sorted(set(p[1] for p in positions))
            if len(rs) == 2 and len(cs) == 2:
                expected = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
                if set(positions) == expected:
                    marker_color = color
                    corners = (rs[0], cs[0], rs[1], cs[1])
                    break
    
    if marker_color is not None:
        # Extract region between markers (exclusive of marker row/col)
        r1, c1, r2, c2 = corners
        region = []
        for r in range(r1 + 1, r2):
            row = []
            for c in range(c1 + 1, c2):
                row.append(grid[r][c])
            region.append(row)
        
        # Find colors in region
        region_counts = {}
        for row in region:
            for v in row:
                region_counts[v] = region_counts.get(v, 0) + 1
        
        # Background in region is the global background color
        # Foreground is the other color in the region
        bg_color = global_bg
        fg_color = None
        for c in region_counts:
            if c != bg_color and c != marker_color:
                fg_color = c
                break
        
        if fg_color is None:
            fg_color = bg_color
        
        # Transform: foreground -> marker_color, background stays
        output = []
        for row in region:
            new_row = []
            for v in row:
                if v == fg_color:
                    new_row.append(marker_color)
                else:
                    new_row.append(bg_color)
            output.append(new_row)
        return output
    
    # No corner markers - find blob and return bounding box filled with background
    bg_color = global_bg
    blob_color = sorted_global[1][0] if len(sorted_global) > 1 else None
    
    if blob_color is not None:
        positions = color_positions[blob_color]
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        
        # Check if boundary columns have isolated pixels (count=1)
        col_counts = {}
        for r, c in positions:
            col_counts[c] = col_counts.get(c, 0) + 1
        
        left_isolated = col_counts.get(min_c, 0) == 1
        right_isolated = col_counts.get(max_c, 0) == 1
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        # If boundary columns are isolated, add 1 to width
        if left_isolated or right_isolated:
            width += 1
        
        return [[bg_color] * width for _ in range(height)]
    
    return grid
