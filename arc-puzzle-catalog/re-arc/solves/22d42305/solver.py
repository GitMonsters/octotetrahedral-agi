def transform(grid):
    """
    Pattern: Find rectangular shape with frame and interior fill.
    Where interior fill "pokes through" frame boundary, extend rays to grid edges.
    Fill the interior with the frame color.
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find background color (most common, typically at corners)
    bg = grid[0, 0]
    
    # Find all non-background pixels
    non_bg_mask = grid != bg
    if not non_bg_mask.any():
        return grid.tolist()
    
    rows, cols = np.where(non_bg_mask)
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    
    # Extract the rectangular region
    region = grid[min_r:max_r+1, min_c:max_c+1]
    
    # Find the two colors in the region (frame and fill)
    colors = set(region.flatten()) - {bg}
    if len(colors) == 1:
        # Only one non-bg color - the fill is same as background
        frame_color = list(colors)[0]
        fill_color = bg
    else:
        # Two colors - need to determine which is frame, which is fill
        colors = list(colors)
        # Frame color forms the boundary, fill is interior
        # Count which color appears more on the outer edge of region
        edge_pixels = list(region[0, :]) + list(region[-1, :]) + list(region[1:-1, 0]) + list(region[1:-1, -1])
        edge_counts = {c: edge_pixels.count(c) for c in colors}
        frame_color = max(colors, key=lambda c: edge_counts.get(c, 0))
        fill_color = [c for c in colors if c != frame_color][0]
    
    # Create output - start with input
    out = grid.copy()
    
    # Find where fill color appears at the boundary of the frame region
    # and extend rays outward
    
    # Check each row in region for fill color at left/right boundaries
    for r in range(min_r, max_r + 1):
        row_data = grid[r, min_c:max_c+1]
        non_bg_in_row = np.where(row_data != bg)[0]
        if len(non_bg_in_row) == 0:
            continue
        
        left_idx = non_bg_in_row[0] + min_c
        right_idx = non_bg_in_row[-1] + min_c
        
        # Left boundary - if fill color
        if grid[r, left_idx] == fill_color:
            # Extend ray left to edge
            out[r, 0:left_idx] = fill_color
            # Extend ray right to edge
            out[r, right_idx+1:w] = fill_color
        # Right boundary - if fill color  
        elif grid[r, right_idx] == fill_color:
            out[r, 0:left_idx] = fill_color
            out[r, right_idx+1:w] = fill_color
    
    # Check each column for fill color at top/bottom boundaries
    for c in range(min_c, max_c + 1):
        col_data = grid[min_r:max_r+1, c]
        non_bg_in_col = np.where(col_data != bg)[0]
        if len(non_bg_in_col) == 0:
            continue
        
        top_idx = non_bg_in_col[0] + min_r
        bot_idx = non_bg_in_col[-1] + min_r
        
        # Top boundary - if fill color
        if grid[top_idx, c] == fill_color:
            # Extend ray up to edge
            out[0:top_idx, c] = fill_color
            # Extend ray down to edge
            out[bot_idx+1:h, c] = fill_color
        # Bottom boundary - if fill color
        elif grid[bot_idx, c] == fill_color:
            out[0:top_idx, c] = fill_color
            out[bot_idx+1:h, c] = fill_color
    
    # Fill interior with frame color (the region becomes "hollow")
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r, c] == fill_color:
                out[r, c] = frame_color
    
    return out.tolist()


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['22d42305']
    
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            import numpy as np
            res_arr = np.array(result)
            exp_arr = np.array(expected)
            if res_arr.shape != exp_arr.shape:
                print(f"  Shape mismatch: {res_arr.shape} vs {exp_arr.shape}")
            else:
                diff = np.where(res_arr != exp_arr)
                print(f"  Differences at {len(diff[0])} positions")
                for r, c in zip(diff[0][:5], diff[1][:5]):
                    print(f"    ({r},{c}): got {res_arr[r,c]}, expected {exp_arr[r,c]}")
