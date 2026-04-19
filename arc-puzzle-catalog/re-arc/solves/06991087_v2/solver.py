import copy


def transform(grid):
    """
    Transform by swapping colors in rectangular regions.
    
    The pattern identifies:
    1. A static color that doesn't change
    2. Two other colors that swap
    3. Regions where the swap happens based on spatial boundaries
    """
    result = copy.deepcopy(grid)
    colors = sorted(set(c for row in grid for c in row))
    
    if len(colors) != 3:
        return result
    
    # Find which color is static by analyzing the structure
    # The static color is typically:
    # - The rarest, forming a compact rectangular region
    # - OR the one marked as a thin line
    
    # First, try to find a marker (thin line)
    marker_color = None
    marker_cols_set = None
    is_vertical_marker = False
    
    for test_color in colors:
        positions = set((r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == test_color)
        rows = set(r for r, c in positions)
        cols = set(c for r, c in positions)
        
        if len(cols) <= 2 and len(rows) > max(1, len(grid) // 2):
            marker_color = test_color
            marker_cols_set = cols
            is_vertical_marker = True
            break
        elif len(rows) <= 2 and len(cols) > max(1, len(grid[0]) // 2):
            marker_color = test_color
            is_vertical_marker = False
            break
    
    # If marker found, use marker-based logic
    if marker_color:
        non_marker = [c for c in colors if c != marker_color]
        if len(non_marker) != 2:
            return result
        
        color1, color2 = non_marker
        
        # Find bounding boxes
        def get_bbox(color):
            positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == color]
            if not positions:
                return None
            rs = [r for r, c in positions]
            cs = [c for r, c in positions]
            return (min(rs), max(rs), min(cs), max(cs))
        
        bbox1 = get_bbox(color1)
        bbox2 = get_bbox(color2)
        
        if not bbox1 or not bbox2:
            return result
        
        size1 = (bbox1[3] - bbox1[2] + 1) * (bbox1[1] - bbox1[0] + 1)
        size2 = (bbox2[3] - bbox2[2] + 1) * (bbox2[1] - bbox2[0] + 1)
        
        if size1 < size2:
            special_color, spec_bbox = color1, bbox1
            outside_color = color2
        else:
            special_color, spec_bbox = color2, bbox2
            outside_color = color1
        
        spec_r_min, spec_r_max, spec_c_min, spec_c_max = spec_bbox
        
        if is_vertical_marker:
            # Find first row where marker appears in special region
            first_marker_in_special = None
            for r in range(spec_r_min, spec_r_max + 1):
                for c in marker_cols_set:
                    if grid[r][c] == marker_color:
                        first_marker_in_special = r
                        break
                if first_marker_in_special is not None:
                    break
            
            if first_marker_in_special is None:
                first_marker_in_special = spec_r_max + 1
            
            # Swap in columns of special region
            for r in range(len(result)):
                for c in range(spec_c_min, spec_c_max + 1):
                    if result[r][c] == marker_color:
                        continue
                    
                    if r < spec_r_min or (r >= first_marker_in_special and r <= spec_r_max):
                        if r < spec_r_min:
                            if result[r][c] == outside_color:
                                result[r][c] = special_color
                        else:
                            if result[r][c] == special_color:
                                result[r][c] = outside_color
        else:
            # Horizontal marker - similar logic
            pass
        
        return result
    
    # No marker - find static color and swap the other two
    # Static color is the rarest one
    counts = {c: sum(row.count(c) for row in grid) for c in colors}
    static_color = min(colors, key=lambda c: counts[c])
    swap_colors = [c for c in colors if c != static_color]
    
    if len(swap_colors) != 2:
        return result
    
    c1, c2 = swap_colors
    
    # Find columns/rows that contain the static color
    static_cols = set()
    static_rows = set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == static_color:
                static_cols.add(c)
                static_rows.add(r)
    
    # Get bounding box of swap colors to know where to swap
    def get_bbox(color):
        positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == color]
        if not positions:
            return None
        rs = [r for r, c in positions]
        cs = [c for r, c in positions]
        return (min(rs), max(rs), min(cs), max(cs))
    
    bbox_c1 = get_bbox(c1)
    bbox_c2 = get_bbox(c2)
    
    if not bbox_c1 or not bbox_c2:
        return result
    
    # Determine which is "special" (smaller region)
    size1 = (bbox_c1[3] - bbox_c1[2] + 1) * (bbox_c1[1] - bbox_c1[0] + 1)
    size2 = (bbox_c2[3] - bbox_c2[2] + 1) * (bbox_c2[1] - bbox_c2[0] + 1)
    
    if size1 < size2:
        special_color, spec_bbox = c1, bbox_c1
        outside_color = c2
    else:
        special_color, spec_bbox = c2, bbox_c2
        outside_color = c1
    
    spec_r_min, spec_r_max, spec_c_min, spec_c_max = spec_bbox
    
    # Swap in the ROWS of special region, skipping cells with static color
    for r in range(spec_r_min, spec_r_max + 1):
        for c in range(len(result[r])):
            if result[r][c] == static_color:
                continue  # Skip cells with static color
            
            # Swap non-static colors in this row
            if result[r][c] == special_color:
                result[r][c] = outside_color
            elif result[r][c] == outside_color:
                result[r][c] = special_color
    
    return result
