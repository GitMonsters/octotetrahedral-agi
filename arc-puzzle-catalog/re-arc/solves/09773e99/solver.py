def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Find background
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Find non-bg cells
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]
    
    # Group by connectivity
    def flood(cells, start):
        visited = {start}
        queue = [start]
        while queue:
            rr, cc = queue.pop()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = rr + dr, cc + dc
                    if (nr, nc) in cells and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return visited
    
    remaining = set(non_bg.keys())
    groups = []
    while remaining:
        start = next(iter(remaining))
        group = flood(remaining, start)
        groups.append(group)
        remaining -= group
    
    # Find template (multi-color group)
    template = None
    template_colors = set()
    for g in groups:
        colors = set(non_bg[p] for p in g)
        if len(colors) >= 2:
            if template is None or len(g) > len(template):
                template = g
                template_colors = colors
    
    if not template:
        return out
    
    # Template bounding box
    t_min_r = min(r for r, c in template)
    t_max_r = max(r for r, c in template)
    t_min_c = min(c for r, c in template)
    t_max_c = max(c for r, c in template)
    
    # For each color in template, find its position and use as reference
    color_to_pos = {}
    for p in template:
        c = non_bg[p]
        if c not in color_to_pos:
            color_to_pos[c] = p
    
    # Template relative to each color's reference position
    template_by_color = {}
    for ref_color, ref_pos in color_to_pos.items():
        template_rel = {}
        for p in template:
            dr, dc = p[0] - ref_pos[0], p[1] - ref_pos[1]
            template_rel[(dr, dc)] = non_bg[p]
        template_by_color[ref_color] = template_rel
    
    # Find external markers (single cells with colors from template)
    markers = []
    for g in groups:
        if len(g) == 1:
            p = next(iter(g))
            color = non_bg[p]
            if color in template_colors and p not in template:
                markers.append((p, color))
    
    # For each marker, place template with appropriate flip
    for (mr, mc), marker_color in markers:
        # Use the template relative to this marker's color
        template_rel = template_by_color.get(marker_color)
        if not template_rel:
            continue
        
        # Determine if marker is left/right/above/below template bbox
        left_of = mc < t_min_c
        right_of = mc > t_max_c
        above = mr < t_min_r
        below = mr > t_max_r
        
        # Apply transformation
        for (dr, dc), v in template_rel.items():
            new_dr, new_dc = dr, dc
            
            # Horizontal flip if marker is to the left
            if left_of:
                new_dc = -new_dc
            
            # Vertical flip if marker is above
            if above:
                new_dr = -new_dr
            
            nr, nc = mr + new_dr, mc + new_dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr][nc] = v
    
    return out
