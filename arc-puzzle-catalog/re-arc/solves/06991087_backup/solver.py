import copy


def transform(grid):
    """
    The transformation pattern:
    1. If 3 colors: identify which color is static (doesn't swap)
    2. The other two colors swap, but ONLY outside columns/rows where static color appears
    3. OR if there's a marker line, use marker-based spatial swapping
    """
    """
    Transform by swapping colors in regions separated by a marker line.
    
    The pattern:
    1. If 3 colors: Find a marker color (thin line), or identify static color
    2. Find rectangular regions
    3. Swap colors based on region boundaries and marker position
    """
    result = copy.deepcopy(grid)
    colors = sorted(set(c for row in grid for c in row))
    
    # No transformation for 2-color or 1-color grids
    if len(colors) != 3:
        return result
    
    counts = {c: sum(row.count(c) for row in grid) for c in colors}
    
    # Helper to get bounding box
    def get_bbox(color):
        min_r, max_r = len(grid), -1
        min_c, max_c = len(grid[0]), -1
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        return (min_r, max_r, min_c, max_c) if max_r >= 0 else None
    
    # Find marker (thin line)
    marker_color = None
    marker_cols = None
    marker_rows = None
    is_vertical = False
    
    for test_color in colors:
        positions = set((r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == test_color)
        rows = set(r for r, c in positions)
        cols = set(c for r, c in positions)
        
        if len(cols) <= 2 and len(rows) > max(1, len(grid) // 2):
            marker_color = test_color
            marker_cols = sorted(cols)
            is_vertical = True
            break
        elif len(rows) <= 2 and len(cols) > max(1, len(grid[0]) // 2):
            marker_color = test_color
            marker_rows = sorted(rows)
            is_vertical = False
            break
    
    # If no marker found, identify static color differently
    if marker_color is None:
        # The static color is likely the smallest/rarest one that doesn't swap
        # For now, swap the two non-dominant colors
        sorted_by_count = sorted(colors, key=lambda c: counts[c])
        
        # Try: identify which of the two smaller colors is static
        c1, c2, background = sorted_by_count[0], sorted_by_count[1], sorted_by_count[2]
        
        # Get bboxes to identify static (usually more compact)
        bbox_c1 = get_bbox(c1)
        bbox_c2 = get_bbox(c2)
        
        if bbox_c1 and bbox_c2:
            size_c1 = (bbox_c1[3] - bbox_c1[2] + 1) * (bbox_c1[1] - bbox_c1[0] + 1)
            size_c2 = (bbox_c2[3] - bbox_c2[2] + 1) * (bbox_c2[1] - bbox_c2[0] + 1)
            
            # Smaller region is static, larger one swaps with background
            if size_c1 < size_c2:
                # c1 is static, swap c2 and background
                marker_color = c1
                swap_color1, swap_color2 = c2, background
            else:
                # c2 is static, swap c1 and background
                marker_color = c2
                swap_color1, swap_color2 = c1, background
        else:
            # Fallback: swap two smallest by count
            swap_color1, swap_color2 = c1, background
        
        # Find regions and apply region-based swap
        colors_to_process = [swap_color1, swap_color2]
        bbox1 = get_bbox(swap_color1)
        bbox2 = get_bbox(swap_color2)
        
        if not bbox1 or not bbox2:
            return result
        
        size1 = (bbox1[3] - bbox1[2] + 1) * (bbox1[1] - bbox1[0] + 1)
        size2 = (bbox2[3] - bbox2[2] + 1) * (bbox2[1] - bbox2[0] + 1)
        
        if size1 < size2:
            special_color, special_bbox = swap_color1, bbox1
            outside_color = swap_color2
        else:
            special_color, special_bbox = swap_color2, bbox2
            outside_color = swap_color1
        
        spec_r_min, spec_r_max, spec_c_min, spec_c_max = special_bbox
        
        # Apply spatial swap (vertical-like logic, since no clear marker direction)
        for r in range(spec_r_min, spec_r_max + 1):  # Only within special region rows
            for c in range(spec_c_min, spec_c_max + 1):
                if r < spec_r_min:
                    if result[r][c] == outside_color:
                        result[r][c] = special_color
                elif r <= spec_r_max:
                    if result[r][c] == special_color:
                        result[r][c] = outside_color
        
        return result
    
    # Marker found - process with marker logic
    non_marker = [c for c in colors if c != marker_color]
    if len(non_marker) != 2:
        return result
    
    color1, color2 = non_marker
    
    bbox1 = get_bbox(color1)
    bbox2 = get_bbox(color2)
    
    if not bbox1 or not bbox2:
        return result
    
    # Identify special (smaller/rarer) color
    size1 = (bbox1[3] - bbox1[2] + 1) * (bbox1[1] - bbox1[0] + 1)
    size2 = (bbox2[3] - bbox2[2] + 1) * (bbox2[1] - bbox2[0] + 1)
    
    if size1 < size2:
        special_color, special_bbox = color1, bbox1
        outside_color = color2
    else:
        special_color, special_bbox = color2, bbox2
        outside_color = color1
    
    spec_r_min, spec_r_max, spec_c_min, spec_c_max = special_bbox
    
    if is_vertical:
        # Find first row where marker appears within special region
        first_marker_in_special = None
        for r in range(spec_r_min, spec_r_max + 1):
            for mc in marker_cols:
                if grid[r][mc] == marker_color:
                    first_marker_in_special = r
                    break
            if first_marker_in_special is not None:
                break
        
        if first_marker_in_special is None:
            first_marker_in_special = spec_r_max + 1
        
        # Swap in the columns of the special region
        for r in range(len(result)):
            for c in range(spec_c_min, spec_c_max + 1):
                if result[r][c] == marker_color:
                    continue
                
                # Swap if before special or from first marker onwards
                should_swap = (r < spec_r_min) or (r >= first_marker_in_special and r <= spec_r_max)
                
                if should_swap:
                    if r < spec_r_min:
                        if result[r][c] == outside_color:
                            result[r][c] = special_color
                    else:
                        if result[r][c] == special_color:
                            result[r][c] = outside_color
    
    else:
        # Horizontal marker
        first_marker_in_special = None
        for c in range(spec_c_min, spec_c_max + 1):
            for mr in marker_rows:
                if grid[mr][c] == marker_color:
                    first_marker_in_special = c
                    break
            if first_marker_in_special is not None:
                break
        
        if first_marker_in_special is None:
            first_marker_in_special = spec_c_max + 1
        
        # Swap in rows of special region
        for r in range(len(result)):
            for c in range(len(result[r])):
                if result[r][c] == marker_color:
                    continue
                
                should_swap = (c < spec_c_min) or (c >= first_marker_in_special and c <= spec_c_max)
                
                if should_swap:
                    if c < spec_c_min:
                        if result[r][c] == outside_color:
                            result[r][c] = special_color
                    else:
                        if result[r][c] == special_color:
                            result[r][c] = outside_color
    
    return result
