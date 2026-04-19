def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    # Find background color
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Find all non-bg cells
    non_bg = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != bg:
                non_bg[(r, c)] = v
    
    # Group by connectivity (8-connected)
    def flood(cells, start):
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
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
    
    # Find templates (groups with 2+ colors)
    templates = []
    for g in groups:
        colors = set(non_bg[p] for p in g)
        if len(colors) >= 2:
            min_r = min(r for r, c in g)
            min_c = min(c for r, c in g)
            pattern = {}
            for r, c in g:
                pattern[(r - min_r, c - min_c)] = non_bg[(r, c)]
            templates.append((colors, pattern, g))
    
    # Mark template cells
    template_cells = set()
    for _, _, g in templates:
        template_cells.update(g)
    
    # Find single-color groups (potential incomplete copies)
    single_color_groups = []
    for g in groups:
        colors = set(non_bg[p] for p in g)
        if len(colors) == 1:
            color = non_bg[next(iter(g))]
            if not (g & template_cells):  # Not part of a template
                single_color_groups.append((color, g))
    
    # Group isolated single cells by color
    isolated_by_color = {}
    multi_cell_groups = []
    for color, group in single_color_groups:
        if len(group) == 1:
            if color not in isolated_by_color:
                isolated_by_color[color] = []
            isolated_by_color[color].append(next(iter(group)))
        else:
            multi_cell_groups.append((color, group))
    
    # For multi-cell groups, match to template
    for color, group in multi_cell_groups:
        matching_templates = [t for t in templates if color in t[0]]
        for template_colors, pattern, _ in matching_templates:
            template_color_cells = [(r, c) for (r, c), v in pattern.items() if v == color]
            if not template_color_cells:
                continue
            
            t_min_r = min(r for r, c in template_color_cells)
            t_min_c = min(c for r, c in template_color_cells)
            template_color_rel = frozenset((r - t_min_r, c - t_min_c) for r, c in template_color_cells)
            
            group_list = list(group)
            min_gr = min(r for r, c in group_list)
            min_gc = min(c for r, c in group_list)
            group_rel = frozenset((r - min_gr, c - min_gc) for r, c in group_list)
            
            if group_rel == template_color_rel:
                template_origin_r = min_gr - t_min_r
                template_origin_c = min_gc - t_min_c
                for (pr, pc), pv in pattern.items():
                    if pv != color:
                        nr, nc = pr + template_origin_r, pc + template_origin_c
                        if 0 <= nr < H and 0 <= nc < W:
                            out[nr][nc] = pv
                break
    
    # For isolated single cells, check if they match template patterns
    # CRITICAL: Sort templates by # of cells of target color (descending)
    # This ensures multi-cell patterns are matched before single-cell patterns
    for color, positions in isolated_by_color.items():
        matching_templates = [t for t in templates if color in t[0]]
        if not matching_templates:
            continue
        
        # Sort by number of color cells (descending) - match larger patterns first
        def count_color_cells(tmpl):
            _, pattern, _ = tmpl
            return len([v for v in pattern.values() if v == color])
        matching_templates = sorted(matching_templates, key=count_color_cells, reverse=True)
        
        # Convert positions to set for faster lookup
        pos_set = set(positions)
        used = set()
        
        for template_colors, pattern, _ in matching_templates:
            # Get template's cells of this color
            template_color_cells = [(r, c) for (r, c), v in pattern.items() if v == color]
            if not template_color_cells:
                continue
            
            # Normalize template color pattern
            t_min_r = min(r for r, c in template_color_cells)
            t_min_c = min(c for r, c in template_color_cells)
            
            # Try to find groups of isolated cells matching this pattern
            for pos in positions:
                if pos in used:
                    continue
                
                # Assume this pos corresponds to (t_min_r, t_min_c) in template
                origin_r = pos[0] - t_min_r
                origin_c = pos[1] - t_min_c
                
                # Check if all template color cells exist
                expected = set()
                for (tr, tc) in template_color_cells:
                    expected.add((tr - t_min_r + pos[0], tc - t_min_c + pos[1]))
                
                if expected.issubset(pos_set) and not (expected & used):
                    # Found a match! Add other colors
                    used.update(expected)
                    for (pr, pc), pv in pattern.items():
                        if pv != color:
                            nr = pr + origin_r
                            nc = pc + origin_c
                            if 0 <= nr < H and 0 <= nc < W:
                                out[nr][nc] = pv
    
    return out
