from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    # Find 8-connected clusters
    visited = [[False]*cols for _ in range(rows)]
    clusters = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg or visited[r][c]: continue
            q = [(r,c)]
            visited[r][c] = True
            cells = []
            while q:
                cr, cc = q.pop(0)
                cells.append((cr, cc, grid[cr][cc]))
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]!=bg:
                            visited[nr][nc] = True
                            q.append((nr,nc))
            clusters.append(cells)
    
    def normalize(cl):
        rs = [r for r,c,_ in cl]
        cs = [c for r,c,_ in cl]
        mr, mc = min(rs), min(cs)
        return tuple(sorted((r-mr, c-mc, v) for r,c,v in cl))
    
    # Separate multi-color templates from mono-color clusters
    templates = []
    mono_clusters = []
    template_cells = set()
    for cl in clusters:
        colors = set(v for _,_,v in cl)
        if len(colors) >= 2:
            templates.append(cl)
            for r,c,_ in cl:
                template_cells.add((r,c))
        else:
            mono_clusters.append(cl)
    
    if not templates:
        return grid
    
    norm_templates = [normalize(cl) for cl in templates]
    
    result = [row[:] for row in grid]
    used = set(template_cells)
    
    # Build all valid placements
    def find_placements(tmpl):
        placements = []
        tmpl_h = max(dr for dr,dc,v in tmpl) + 1
        tmpl_w = max(dc for dr,dc,v in tmpl) + 1
        for ar in range(rows - tmpl_h + 1):
            for ac in range(cols - tmpl_w + 1):
                matches = 0
                conflicts = 0
                new_cells = []
                match_cells = []
                for dr, dc, v in tmpl:
                    nr, nc = ar+dr, ac+dc
                    cur = result[nr][nc]
                    if cur == v:
                        matches += 1
                        match_cells.append((nr,nc))
                    elif cur == bg:
                        new_cells.append((nr,nc,v))
                    else:
                        conflicts += 1
                if conflicts == 0 and new_cells and matches >= 1:
                    fresh = [c for c in match_cells if c not in used]
                    if fresh:
                        placements.append((matches, ar, ac, new_cells, match_cells, set(match_cells)))
        return placements
    
    all_placements = []
    for tmpl in norm_templates:
        for p in find_placements(tmpl):
            all_placements.append(p + (tmpl,))
    
    # Sort: most matches first
    all_placements.sort(key=lambda x: -x[0])
    
    # Greedy assignment with cluster-shape awareness
    # First pass: placements with matches >= 2
    applied = []
    for p in all_placements:
        matches, ar, ac, new_cells, match_cells, match_set, tmpl = p
        if matches < 2:
            continue
        # Check still valid
        ok = True
        fresh = [c for c in match_cells if c not in used]
        if not fresh:
            continue
        for nr,nc,v in new_cells:
            if result[nr][nc] != bg:
                ok = False; break
        if not ok:
            continue
        for nr,nc,v in new_cells:
            result[nr][nc] = v
        for c in match_cells:
            used.add(c)
        for nr,nc,_ in new_cells:
            used.add((nr,nc))
        applied.append(p)
    
    # Second pass: placements with matches == 1
    # For each unused mono-cluster, find its best placement
    remaining = []
    for cl in mono_clusters:
        cl_cells = set((r,c) for r,c,_ in cl)
        if cl_cells & used:
            continue
        remaining.append(cl)
    
    for cl in remaining:
        cl_cells = set((r,c) for r,c,_ in cl)
        cl_color = cl[0][2]
        cl_size = len(cl)
        
        best = None
        best_score = -1
        
        for p in all_placements:
            matches, ar, ac, new_cells, match_cells, match_set, tmpl = p
            # Check if this cluster's cells are among the match cells
            overlap = cl_cells & match_set
            if not overlap:
                continue
            # Check still valid
            ok = True
            actual_new = []
            actual_match = []
            for dr, dc, v in tmpl:
                nr, nc = ar+dr, ac+dc
                if result[nr][nc] == v:
                    actual_match.append((nr,nc))
                elif result[nr][nc] == bg:
                    actual_new.append((nr,nc,v))
                else:
                    ok = False; break
            if not ok or not actual_new:
                continue
            fresh = [c for c in actual_match if c not in used]
            if not fresh:
                continue
            
            # Score: count how many of this cluster's cells match same-color template positions
            same_color_positions = sum(1 for dr,dc,v in tmpl if v == cl_color)
            overlap_count = len(overlap)
            # Prefer placements where cluster covers ALL same-color positions
            score = overlap_count * 100 + (100 if overlap_count == same_color_positions else 0)
            
            if score > best_score:
                best_score = score
                best = (actual_new, actual_match)
        
        if best:
            actual_new, actual_match = best
            for nr,nc,v in actual_new:
                if result[nr][nc] == bg:
                    result[nr][nc] = v
            for c in actual_match:
                used.add(c)
            for nr,nc,_ in actual_new:
                used.add((nr,nc))
    
    return result
