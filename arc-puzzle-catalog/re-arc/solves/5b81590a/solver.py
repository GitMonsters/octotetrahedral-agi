def transform(grid):
    from collections import Counter
    
    # Determine background color (most common)
    all_vals = []
    for row in grid:
        all_vals.extend(row)
    bg_color = Counter(all_vals).most_common(1)[0][0]
    
    # Find non-background pixels and identify rectangle vs scattered
    pixels_by_value = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            val = grid[r][c]
            if val != bg_color:
                if val not in pixels_by_value:
                    pixels_by_value[val] = []
                pixels_by_value[val].append((r, c))
    
    # Start with output as copy of input
    output = [row[:] for row in grid]
    
    # For each non-background value, check if it's a rectangle or scattered pixels
    rect_color = None
    rect_bounds = None
    scattered_color = None
    scattered_positions = []
    
    for val, positions in pixels_by_value.items():
        if not positions:
            continue
        
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        
        rect_size = (max_r - min_r + 1) * (max_c - min_c + 1)
        
        if len(positions) == rect_size:
            # It's a solid rectangle
            rect_color = val
            rect_bounds = (min_r, max_r, min_c, max_c)
        else:
            # It's scattered pixels
            scattered_color = val
            scattered_positions = positions
    
    # If no scattered pixels, return input as-is
    if not scattered_positions or rect_bounds is None:
        return output
    
    min_r, max_r, min_c, max_c = rect_bounds
    
    # Generate border positions (1 cell away from rectangle)
    border_positions = set()
    # Top border
    for c in range(min_c - 1, max_c + 2):
        border_positions.add((min_r - 1, c))
    # Bottom border
    for c in range(min_c - 1, max_c + 2):
        border_positions.add((max_r + 1, c))
    # Left border (excluding corners already added)
    for r in range(min_r, max_r + 1):
        border_positions.add((r, min_c - 1))
    # Right border (excluding corners already added)
    for r in range(min_r, max_r + 1):
        border_positions.add((r, max_c + 1))
    
    border_positions = list(border_positions)
    
    # Clear scattered pixels from output
    for r, c in scattered_positions:
        output[r][c] = bg_color
    
    # Move each scattered pixel to nearest border position
    for in_pos in scattered_positions:
        # Find nearest border position
        min_dist = float('inf')
        nearest_border = None
        
        for border_pos in border_positions:
            dist_sq = (in_pos[0] - border_pos[0]) ** 2 + (in_pos[1] - border_pos[1]) ** 2
            if dist_sq < min_dist:
                min_dist = dist_sq
                nearest_border = border_pos
        
        if nearest_border is not None:
            output[nearest_border[0]][nearest_border[1]] = scattered_color
    
    return output
