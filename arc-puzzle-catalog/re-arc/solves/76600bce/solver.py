from collections import Counter, deque

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    cnt = Counter()
    for r in grid:
        for c in r:
            cnt[c] += 1
    
    bg = cnt.most_common(1)[0][0]
    non_bg_colors = [k for k in cnt if k != bg]
    
    if len(non_bg_colors) < 2:
        return [row[:] for row in grid]
    
    # Find connected components of non-bg cells
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in visited:
                comp = {}
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp[(cr, cc)] = grid[cr][cc]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and grid[nr][nc] != bg:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)
    
    # Find the template: component with multiple non-bg colors
    template_comp = None
    core_color = None
    halo_color = None
    
    for comp in components:
        colors_in_comp = set(comp.values())
        if len(colors_in_comp) >= 2:
            template_comp = comp
            break
    
    if template_comp is None:
        return [row[:] for row in grid]
    
    # Determine core and halo colors
    # Core = color that appears in other components too
    # Halo = color only in template
    template_colors = set(template_comp.values())
    other_colors = set()
    for comp in components:
        if comp is not template_comp:
            other_colors.update(comp.values())
    
    for tc in template_colors:
        if tc in other_colors:
            core_color = tc
        else:
            halo_color = tc
    
    if core_color is None or halo_color is None:
        return [row[:] for row in grid]
    
    # Extract template bounding box
    t_cells = list(template_comp.keys())
    t_min_r = min(r for r, c in t_cells)
    t_max_r = max(r for r, c in t_cells)
    t_min_c = min(c for r, c in t_cells)
    t_max_c = max(c for r, c in t_cells)
    
    th = t_max_r - t_min_r + 1
    tw = t_max_c - t_min_c + 1
    
    # Build template grid (with bg for empty cells)
    template = [[bg] * tw for _ in range(th)]
    for (r, c), v in template_comp.items():
        template[r - t_min_r][c - t_min_c] = v
    
    result = [row[:] for row in grid]
    
    # Try multiple scale factors
    max_scale = max(rows, cols)
    for s in range(1, max_scale + 1):
        scaled_h = th * s
        scaled_w = tw * s
        if scaled_h > rows and scaled_w > cols:
            break
        
        # Try all positions
        for sr in range(rows - scaled_h + 1):
            for sc in range(cols - scaled_w + 1):
                # Check if scaled template matches here
                valid = True
                for tr in range(th):
                    if not valid:
                        break
                    for tc in range(tw):
                        tv = template[tr][tc]
                        # Check s×s block
                        for dr in range(s):
                            if not valid:
                                break
                            for dc in range(s):
                                gr = sr + tr * s + dr
                                gc = sc + tc * s + dc
                                gv = grid[gr][gc]
                                if tv == core_color:
                                    if gv != core_color:
                                        valid = False
                                        break
                                elif tv == halo_color:
                                    if gv != bg:
                                        valid = False
                                        break
                                # tv == bg: anything is ok (we skip)
                
                if valid:
                    # Place halo cells
                    for tr in range(th):
                        for tc in range(tw):
                            if template[tr][tc] == halo_color:
                                for dr in range(s):
                                    for dc in range(s):
                                        gr = sr + tr * s + dr
                                        gc = sc + tc * s + dc
                                        if result[gr][gc] == bg:
                                            result[gr][gc] = halo_color
    
    return result
