
from collections import Counter, deque, defaultdict
import copy

def transform(grid):
    """
    ARC puzzle 38727a99 solver.
    
    Handles two types of puzzles:
    1. TRAIN 0 style: 2-value template + single-cell probes
       - Uses constraint satisfaction with backtracking
    2. TRAIN 1/2 style: Multi-value template + multi-cell probes
       - Completes partial templates with best-match rotation
    """
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    def get_bg(g):
        return Counter(c for row in g for c in row).most_common(1)[0][0]
    
    def get_components(g, background):
        visited = set()
        components = []
        for r in range(h):
            for c in range(w):
                if g[r][c] != background and (r,c) not in visited:
                    comp = {}
                    queue = deque([(r,c)])
                    visited.add((r,c))
                    while queue:
                        cr, cc = queue.popleft()
                        comp[(cr,cc)] = g[cr][cc]
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and g[nr][nc] != background:
                                visited.add((nr,nc))
                                queue.append((nr,nc))
                    components.append(comp)
        return components
    
    def apply_sym(dr, dc, sym):
        transforms = [
            (dr, dc), (dc, -dr), (-dr, -dc), (-dc, dr),
            (dr, -dc), (-dr, dc), (dc, dr), (-dc, -dr),
        ]
        return transforms[sym]
    
    bg = get_bg(grid)
    comps = get_components(grid, bg)
    
    template_comp = max(comps, key=lambda c: len(set(c.values())))
    template_values = set(template_comp.values())
    
    result = copy.deepcopy(grid)
    
    single_comps = [c for c in comps if len(c) == 1]
    
    # TRAIN 0 style: exactly 2 values in template
    if len(template_values) == 2 and len(single_comps) > 0:
        single_vals = Counter(list(c.values())[0] for c in single_comps)
        
        probe_val = None
        for v in template_values:
            if v in single_vals and (probe_val is None or single_vals[v] > single_vals.get(probe_val, 0)):
                probe_val = v
        
        if probe_val is None:
            return result
        
        satellite_val = [v for v in template_values if v != probe_val][0]
        
        template_probe_pos = None
        template_satellite_pos = None
        for pos, val in template_comp.items():
            if val == probe_val:
                template_probe_pos = pos
            elif val == satellite_val:
                template_satellite_pos = pos
        
        if template_probe_pos is None or template_satellite_pos is None:
            return result
        
        offset1 = (template_satellite_pos[0] - template_probe_pos[0],
                   template_satellite_pos[1] - template_probe_pos[1])
        
        isolated_satellite = None
        for c in single_comps:
            pos, val = list(c.items())[0]
            if val == satellite_val:
                isolated_satellite = pos
                break
        
        if isolated_satellite:
            offset2 = (isolated_satellite[0] - template_probe_pos[0],
                       isolated_satellite[1] - template_probe_pos[1])
        else:
            offset2 = None
        
        probe_cells = []
        for c in single_comps:
            pos, val = list(c.items())[0]
            if val == probe_val and pos != template_probe_pos:
                probe_cells.append(pos)
        
        def get_options(r, c):
            options = []
            for sym in range(8):
                o1 = apply_sym(offset1[0], offset1[1], sym)
                p1 = (r + o1[0], c + o1[1])
                
                if offset2:
                    o2 = apply_sym(offset2[0], offset2[1], sym)
                    p2 = (r + o2[0], c + o2[1])
                    positions = frozenset([p1, p2])
                else:
                    positions = frozenset([p1])
                
                valid = True
                for p in positions:
                    if not (0 <= p[0] < h and 0 <= p[1] < w):
                        valid = False
                        break
                    if grid[p[0]][p[1]] != bg:
                        valid = False
                        break
                
                if valid:
                    options.append((sym, positions))
            return options
        
        domains = {cell: get_options(cell[0], cell[1]) for cell in probe_cells}
        
        pos_to_opts = defaultdict(list)
        for cell, opts in domains.items():
            for sym, positions in opts:
                for p in positions:
                    pos_to_opts[p].append((cell, sym, positions))
        
        assignment = {}
        
        def get_remaining_options(cell, used):
            return [(s, p) for s, p in domains[cell] if not p & used]
        
        def solve():
            used = set()
            
            changed = True
            while changed:
                changed = False
                
                for cell in probe_cells:
                    if cell in assignment:
                        continue
                    
                    opts = get_remaining_options(cell, used)
                    
                    if len(opts) == 0:
                        return False
                    
                    if len(opts) == 1:
                        sym, positions = opts[0]
                        assignment[cell] = positions
                        used.update(positions)
                        changed = True
                
                for p, users in pos_to_opts.items():
                    if p in used:
                        continue
                    
                    available = [(cell, sym, positions) for cell, sym, positions in users
                                if cell not in assignment and not (positions - {p}) & used]
                    
                    if len(available) == 1:
                        cell, sym, positions = available[0]
                        if cell not in assignment:
                            assignment[cell] = positions
                            used.update(positions)
                            changed = True
            
            if len(assignment) == len(probe_cells):
                return True
            
            unassigned = [cell for cell in probe_cells if cell not in assignment]
            unassigned.sort(key=lambda c: len(get_remaining_options(c, used)))
            cell = unassigned[0]
            
            opts = get_remaining_options(cell, used)
            
            for sym, positions in opts:
                old_assignment = dict(assignment)
                old_used = set(used)
                
                assignment[cell] = positions
                used.update(positions)
                
                if solve():
                    return True
                
                assignment.clear()
                assignment.update(old_assignment)
                used.clear()
                used.update(old_used)
            
            return False
        
        if solve():
            for pos, positions in assignment.items():
                for p in positions:
                    result[p[0]][p[1]] = satellite_val
    
    else:
        # TRAIN 1/2 style
        anchor_val = 9 if 9 in template_values else None
        
        template_anchor = None
        if anchor_val:
            for pos, val in template_comp.items():
                if val == anchor_val:
                    template_anchor = pos
                    break
        
        if template_anchor is None:
            template_anchor = list(template_comp.keys())[0]
        
        template = {}
        for (r,c), v in template_comp.items():
            dr, dc = r - template_anchor[0], c - template_anchor[1]
            template[(dr,dc)] = v
        
        template_cells = set(template_comp.keys())
        
        probe_9_positions = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == 9 and (r,c) not in template_cells:
                    probe_9_positions.append((r,c))
        
        if probe_9_positions:
            for ar, ac in probe_9_positions:
                best_sym = None
                best_matches = -1
                
                for sym in range(8):
                    transformed = {}
                    for (dr,dc), v in template.items():
                        new_dr, new_dc = apply_sym(dr, dc, sym)
                        transformed[(new_dr, new_dc)] = v
                    
                    matches = 0
                    valid = True
                    for (dr, dc), v in transformed.items():
                        r = ar + dr
                        c = ac + dc
                        
                        if not (0 <= r < h and 0 <= c < w):
                            valid = False
                            break
                        
                        if (r, c) in template_cells:
                            valid = False
                            break
                        
                        if grid[r][c] != bg:
                            if grid[r][c] == v:
                                matches += 1
                            else:
                                valid = False
                                break
                    
                    if valid and matches > best_matches:
                        best_matches = matches
                        best_sym = sym
                
                if best_sym is not None:
                    transformed = {}
                    for (dr,dc), v in template.items():
                        new_dr, new_dc = apply_sym(dr, dc, best_sym)
                        transformed[(new_dr, new_dc)] = v
                    
                    for (dr, dc), v in transformed.items():
                        r = ar + dr
                        c = ac + dc
                        if grid[r][c] == bg:
                            result[r][c] = v
        else:
            probe_cells = {(r,c): grid[r][c] for r in range(h) for c in range(w) 
                          if grid[r][c] != bg and (r,c) not in template_cells}
            
            valid_placements = []
            
            for ar in range(h):
                for ac in range(w):
                    if grid[ar][ac] != bg:
                        continue
                    
                    for sym in range(8):
                        transformed = {}
                        for (dr,dc), v in template.items():
                            new_dr, new_dc = apply_sym(dr, dc, sym)
                            transformed[(new_dr, new_dc)] = v
                        
                        match = True
                        existing_matches = []
                        
                        for (dr, dc), v in transformed.items():
                            r = ar + dr
                            c = ac + dc
                            
                            if not (0 <= r < h and 0 <= c < w):
                                match = False
                                break
                            
                            if (r, c) in template_cells:
                                match = False
                                break
                            
                            if grid[r][c] != bg:
                                if grid[r][c] != v:
                                    match = False
                                    break
                                existing_matches.append((r, c))
                        
                        if match and len(existing_matches) >= 2:
                            valid_placements.append((ar, ac, sym, set(existing_matches)))
            
            used_cells = set()
            valid_placements.sort(key=lambda x: len(x[3]), reverse=True)
            
            for ar, ac, sym, matches in valid_placements:
                new_matches = matches - used_cells
                
                if len(new_matches) == len(matches) and len(matches) > 0:
                    used_cells.update(matches)
                    
                    transformed = {}
                    for (dr,dc), v in template.items():
                        new_dr, new_dc = apply_sym(dr, dc, sym)
                        transformed[(new_dr, new_dc)] = v
                    
                    for (dr, dc), v in transformed.items():
                        r = ar + dr
                        c = ac + dc
                        if grid[r][c] == bg:
                            result[r][c] = v
    
    return result
