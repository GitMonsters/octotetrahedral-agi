def transform(grid):
    from collections import Counter
    
    H = len(grid)
    W = len(grid[0])
    
    # Find background (most frequent color)
    cc = Counter()
    for row in grid:
        cc.update(row)
    bg = cc.most_common(1)[0][0]
    
    # Find non-bg cells grouped by color
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r, c))
    
    # Shape has more cells, marker has fewer
    colors_sorted = sorted(non_bg.keys(), key=lambda col: len(non_bg[col]))
    if len(colors_sorted) < 2:
        return [row[:] for row in grid]
    marker_color = colors_sorted[0]
    shape_color = colors_sorted[1]
    
    shape_cells = set((r, c) for r, c in non_bg[shape_color])
    marker_cells = set((r, c) for r, c in non_bg[marker_color])
    
    # Shape bounding box
    min_r = min(r for r, c in shape_cells)
    max_r = max(r for r, c in shape_cells)
    min_c = min(c for r, c in shape_cells)
    max_c = max(c for r, c in shape_cells)
    
    # Pattern relative to top-left of bbox
    pattern = set()
    for r, c in shape_cells:
        pattern.add((r - min_r, c - min_c))
    
    # Determine direction from marker positions relative to shape corners
    corners = {
        'TL': (min_r, min_c, -1, -1),
        'TR': (min_r, max_c, -1, +1),
        'BL': (max_r, min_c, +1, -1),
        'BR': (max_r, max_c, +1, +1),
    }
    
    dr, dc = 0, 0
    for name, (cr, cc_val, d_r, d_c) in corners.items():
        adjacent = {(cr + d_r, cc_val), (cr, cc_val + d_c), (cr + d_r, cc_val + d_c)}
        if adjacent.intersection(marker_cells):
            dr, dc = d_r, d_c
            break
    
    # Find minimum step k such that shifted pattern doesn't overlap original
    for k in range(1, max(H, W) + 1):
        overlap = False
        for pr, pc in pattern:
            shifted = (pr + k * dr, pc + k * dc)
            if shifted in pattern:
                overlap = True
                break
        if not overlap:
            break
    
    # Create output (copy input)
    result = [row[:] for row in grid]
    
    # Place copies using marker color
    n = 1
    while True:
        offset_r = min_r + n * k * dr
        offset_c = min_c + n * k * dc
        
        any_on_grid = False
        for pr, pc in pattern:
            nr = offset_r + pr
            nc = offset_c + pc
            if 0 <= nr < H and 0 <= nc < W:
                any_on_grid = True
                result[nr][nc] = marker_color
        
        if not any_on_grid:
            break
        n += 1
    
    return result
