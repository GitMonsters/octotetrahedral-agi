def transform(grid):
    import copy
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find connected components of non-bg cells
    visited = [[False]*cols for _ in range(rows)]
    components = []
    
    def flood_fill(r, c):
        stack = [(r, c)]
        cells = []
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr][cc] or grid[cr][cc] == bg:
                continue
            visited[cr][cc] = True
            cells.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        return cells
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != bg:
                comp = flood_fill(r, c)
                if comp:
                    components.append(comp)
    
    # Classify: single-color large component = template, multi-color = pattern
    template_comp = None
    pattern_comp = None
    
    for comp in components:
        vals = set(grid[r][c] for r,c in comp)
        if len(vals) == 1 and len(comp) > 4:
            if template_comp is None or len(comp) > len(template_comp):
                template_comp = comp
        elif len(vals) > 1:
            if pattern_comp is None or len(comp) > len(pattern_comp):
                pattern_comp = comp
    
    if pattern_comp is None:
        # Fallback: largest multi-color or any non-template component
        for comp in components:
            vals = set(grid[r][c] for r,c in comp)
            if comp != template_comp:
                if pattern_comp is None or len(comp) > len(pattern_comp):
                    pattern_comp = comp
    
    if pattern_comp is None:
        return grid
    
    pr_min = min(r for r,c in pattern_comp)
    pr_max = max(r for r,c in pattern_comp)
    pc_min = min(c for r,c in pattern_comp)
    pc_max = max(c for r,c in pattern_comp)
    orows = pr_max - pr_min + 1
    ocols = pc_max - pc_min + 1
    
    # Extract pattern
    pat = [[grid[pr_min+r][pc_min+c] for c in range(ocols)] for r in range(orows)]
    
    if template_comp is not None:
        # Standard approach: downscale template to create mask
        tr_min = min(r for r,c in template_comp)
        tr_max = max(r for r,c in template_comp)
        tc_min = min(c for r,c in template_comp)
        tc_max = max(c for r,c in template_comp)
        template_color = grid[template_comp[0][0]][template_comp[0][1]]
        
        th = tr_max - tr_min + 1
        tw = tc_max - tc_min + 1
        sr = th // orows
        sc = tw // ocols
        
        mask = []
        for r in range(orows):
            row = []
            for c in range(ocols):
                all_template = True
                for dr in range(sr):
                    for dc in range(sc):
                        rr = tr_min + r*sr + dr
                        cc = tc_min + c*sc + dc
                        if rr >= rows or cc >= cols or grid[rr][cc] != template_color:
                            all_template = False
                row.append(1 if all_template else 0)
            mask.append(row)
    else:
        # Embedded case: no separate template
        # Find the most common non-bg color in the pattern (template color)
        pat_vals = [pat[r][c] for r in range(orows) for c in range(ocols) if pat[r][c] != bg]
        template_color = Counter(pat_vals).most_common(1)[0][0]
        
        # Find template-colored cells on the main diagonal of the pattern
        seeds = set()
        for i in range(min(orows, ocols)):
            if pat[i][i] == template_color:
                seeds.add((i, i))
        
        # Grow seeds by Manhattan distance 1
        mask_set = set()
        for sr, sc in seeds:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if abs(dr) + abs(dc) <= 1:
                        nr, nc = sr + dr, sc + dc
                        if 0 <= nr < orows and 0 <= nc < ocols:
                            mask_set.add((nr, nc))
        
        mask = [[1 if (r,c) in mask_set else 0 for c in range(ocols)] for r in range(orows)]
    
    # Apply mask to pattern
    output = []
    for r in range(orows):
        row = []
        for c in range(ocols):
            if mask[r][c]:
                row.append(pat[r][c])
            else:
                row.append(bg)
        output.append(row)
    
    return output
