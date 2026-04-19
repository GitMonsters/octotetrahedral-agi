def transform(grid):
    """
    ARC puzzle 5fb75ab8 solver.
    
    Core pattern: Find template (anchor + surrounding pattern), then for each 
    other instance of anchor color, stamp the pattern with horizontal reflection.
    """
    import copy
    from collections import Counter, defaultdict, deque
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background
    color_counts = Counter(c for row in grid for c in row)
    background = color_counts.most_common(1)[0][0]
    
    # Find all colors and positions
    color_positions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color_positions[grid[r][c]].append((r, c))
    
    output = copy.deepcopy(grid)
    
    # Find template: cell surrounded by another color's cells
    template_info = None
    max_adjacent = 0
    
    for color in color_positions:
        for r, c in color_positions[color]:
            adjacent = defaultdict(list)
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj_c = grid[nr][nc]
                        if adj_c != background and adj_c != color:
                            adjacent[adj_c].append((dr, dc))
            
            if len(adjacent) == 1:
                adj_color = list(adjacent.keys())[0]
                if len(adjacent[adj_color]) > max_adjacent:
                    max_adjacent = len(adjacent[adj_color])
                    template_info = {
                        'anchor': (r, c),
                        'anchor_color': color,
                        'pattern_color': adj_color,
                        'pattern_offsets': adjacent[adj_color]
                    }
    
    if not template_info:
        return grid
    
    anchor = template_info['anchor']
    anchor_color = template_info['anchor_color']
    pattern_color = template_info['pattern_color']
    pattern_offsets = template_info['pattern_offsets']
    
    # Reflected offsets (horizontal flip)
    reflected = [(dr, -dc) for dr, dc in pattern_offsets]
    
    # Find other anchor-color cells
    def find_clusters(positions, max_dist=2):
        remaining = set(positions)
        clusters = []
        while remaining:
            start = next(iter(remaining))
            cluster = {start}
            q = deque([start])
            while q:
                curr = q.popleft()
                for p in list(remaining - cluster):
                    if abs(p[0]-curr[0]) <= max_dist and abs(p[1]-curr[1]) <= max_dist:
                        cluster.add(p)
                        q.append(p)
            clusters.append(sorted(cluster))
            remaining -= cluster
        return clusters
    
    other_anchors = [p for p in color_positions[anchor_color] if p != anchor]
    clusters = find_clusters(other_anchors, max_dist=2) if other_anchors else []
    
    # Separate isolated cells from shape clusters
    isolated = []
    shape_cluster = None
    for cl in clusters:
        if len(cl) == 1:
            isolated.append(cl[0])
        elif len(cl) >= 3:
            shape_cluster = cl
    
    # Stamp pattern at isolated anchor cells
    for ar, ac in isolated:
        for dr, dc in reflected:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == background:
                output[nr][nc] = pattern_color
    
    # Handle scattered dots (third color)
    closest_pattern = min(color_positions[pattern_color],
                         key=lambda p: abs(p[0]-anchor[0])+abs(p[1]-anchor[1]))
    scatter_offset = (closest_pattern[0] - anchor[0], closest_pattern[1] - anchor[1])
    
    for color in color_positions:
        if color in [background, anchor_color, pattern_color]:
            continue
        positions = color_positions[color]
        pos_set = set(positions)
        
        # Only stamp isolated cells (no adjacent same-color neighbors)
        for sr, sc in positions:
            has_neighbor = any((sr+dr, sc+dc) in pos_set 
                              for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc)
            if not has_neighbor:
                nr, nc = sr + scatter_offset[0], sc + scatter_offset[1]
                if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == background:
                    output[nr][nc] = color
    
    return output
