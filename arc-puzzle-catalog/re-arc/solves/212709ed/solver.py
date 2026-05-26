# TASK: 212709ed
# R1 SCALE:     same size
# R2 STRUCTURE: scattered shapes in one color, small marker in another
# R3 SYMMETRY:  irrelevant
# R4 COLOR:     bg = most frequent, marker = least frequent non-bg, shape = other
# R5 OBJECTS:   Shape cells scattered across grid; marker defines line color
# RULE: For each contiguous group of shape cells, extend by group_size in BOTH directions
# VERIFIED:     train0 ✓   train1 ✓   train2 ✓

from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    # Find background (most frequent color)
    cc = Counter()
    for row in grid:
        cc.update(row)
    bg = cc.most_common(1)[0][0]
    
    # Find all non-bg cells by color
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r, c))
    
    if len(non_bg) == 1:
        # Only one non-bg color
        shape_color = list(non_bg.keys())[0]
        line_color = shape_color
        shape_cells = set(non_bg[shape_color])
    else:
        # Marker is the color with fewer cells, shape is the one with more cells
        colors_sorted = sorted(non_bg.keys(), key=lambda col: len(non_bg[col]))
        marker_color = colors_sorted[0]
        shape_color = colors_sorted[1] if len(colors_sorted) > 1 else colors_sorted[0]
        line_color = marker_color
        shape_cells = set(non_bg[shape_color])
    
    # Create output grid
    output = [row[:] for row in grid]
    
    # Process each row: find contiguous groups and extend in both directions
    for r in range(H):
        groups = []
        current_group = []
        for c in range(W):
            if (r, c) in shape_cells:
                current_group.append(c)
            elif current_group:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)
        
        # Extend each group
        for group in groups:
            first_c = min(group)
            last_c = max(group)
            extend_by = len(group)
            
            # Extend before the group
            for c in range(max(0, first_c - extend_by), first_c):
                if output[r][c] == bg and grid[r][c] == bg:
                    output[r][c] = line_color
            
            # Extend after the group
            for c in range(last_c + 1, min(W, last_c + 1 + extend_by)):
                if output[r][c] == bg and grid[r][c] == bg:
                    output[r][c] = line_color
    
    # Process each column: find contiguous groups and extend in both directions
    for c in range(W):
        groups = []
        current_group = []
        for r in range(H):
            if (r, c) in shape_cells:
                current_group.append(r)
            elif current_group:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)
        
        # Extend each group
        for group in groups:
            first_r = min(group)
            last_r = max(group)
            extend_by = len(group)
            
            # Extend before the group
            for r in range(max(0, first_r - extend_by), first_r):
                if output[r][c] == bg and grid[r][c] == bg:
                    output[r][c] = line_color
            
            # Extend after the group
            for r in range(last_r + 1, min(H, last_r + 1 + extend_by)):
                if output[r][c] == bg and grid[r][c] == bg:
                    output[r][c] = line_color
    
    return output
