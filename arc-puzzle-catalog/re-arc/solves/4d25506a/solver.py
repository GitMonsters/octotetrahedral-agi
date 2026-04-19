from collections import Counter
import copy

def transform(input_grid):
    grid = [row[:] for row in input_grid]
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most common)
    bg = Counter(v for r in grid for v in r).most_common(1)[0][0]
    
    # Find template: largest rectangle of a single non-bg color with all-same borders
    def find_template():
        best = None
        best_area = 0
        vals = set(v for r in grid for v in r) - {bg}
        for v in vals:
            for r1 in range(rows):
                for c1 in range(cols):
                    if grid[r1][c1] != v:
                        continue
                    for r2 in range(r1 + 2, rows):
                        for c2 in range(c1 + 2, cols):
                            h, w = r2 - r1 + 1, c2 - c1 + 1
                            ok = True
                            for rr in range(r1, r2 + 1):
                                if grid[rr][c1] != v or grid[rr][c2] != v:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            for cc in range(c1, c2 + 1):
                                if grid[r1][cc] != v or grid[r2][cc] != v:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            if h * w > best_area:
                                best_area = h * w
                                best = (r1, c1, r2, c2, v)
        return best

    r1, c1, r2, c2, frame_color = find_template()
    h, w = r2 - r1 + 1, c2 - c1 + 1
    template = [[grid[r1 + dr][c1 + dc] for dc in range(w)] for dr in range(h)]
    
    # Determine pattern color (non-bg, non-frame inside template)
    pattern_colors = set()
    for dr in range(h):
        for dc in range(w):
            v = template[dr][dc]
            if v != frame_color and v != bg:
                pattern_colors.add(v)
    
    has_pattern = len(pattern_colors) > 0
    
    # Get all 8 orientations of a template
    def get_orientations(t):
        results = []
        curr = [row[:] for row in t]
        for _ in range(4):
            results.append([row[:] for row in curr])
            results.append([row[::-1] for row in curr])
            th, tw = len(curr), len(curr[0])
            curr = [[curr[th - 1 - r][c] for r in range(th)] for c in range(tw)]
        seen = set()
        unique = []
        for t in results:
            key = tuple(tuple(row) for row in t)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    output = [row[:] for row in grid]
    
    if has_pattern:
        # CASE 1: Template has a distinct pattern color inside
        # Match: frame_color → bg, pattern_color → pattern_color
        # Try all 8 orientations
        for tmpl in get_orientations(template):
            th, tw = len(tmpl), len(tmpl[0])
            for sr in range(rows - th + 1):
                for sc in range(cols - tw + 1):
                    # Skip if overlaps with original template
                    if not (sr > r2 or sr + th - 1 < r1 or sc > c2 or sc + tw - 1 < c1):
                        continue
                    ok = True
                    for dr in range(th):
                        for dc in range(tw):
                            tv = tmpl[dr][dc]
                            gv = grid[sr + dr][sc + dc]
                            if tv == frame_color:
                                if gv != bg:
                                    ok = False
                                    break
                            else:  # pattern color
                                if gv != tv:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        # Stamp: fill bg cells with frame_color
                        for dr in range(th):
                            for dc in range(tw):
                                r, c = sr + dr, sc + dc
                                if output[r][c] == bg:
                                    output[r][c] = frame_color
    else:
        # CASE 2: Template is all one color (frame_color = noise color)
        # Find 8-connected components of noise pixels outside template
        noise_positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == frame_color and not (r1 <= r <= r2 and c1 <= c <= c2):
                    noise_positions.append((r, c))
        
        noise_set = set(noise_positions)
        visited = set()
        components = []
        for p in noise_positions:
            if p in visited:
                continue
            comp = []
            stack = [p]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.append(curr)
                cr, cc = curr
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nb = (cr + dr, cc + dc)
                        if nb in noise_set and nb not in visited:
                            stack.append(nb)
            components.append(comp)
        
        # Interior dimensions
        interior_dims = set()
        interior_dims.add((h - 2, w - 2))
        if h != w:
            interior_dims.add((w - 2, h - 2))
        
        # Check if noise pixels form an independent set (no orthogonal adjacency)
        def is_independent(comp):
            comp_set = set(comp)
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in comp_set:
                        return False
            return True
        
        # Find valid placements
        valid_placements = []
        for comp in components:
            comp_rs = [p[0] for p in comp]
            comp_cs = [p[1] for p in comp]
            bbox_h = max(comp_rs) - min(comp_rs) + 1
            bbox_w = max(comp_cs) - min(comp_cs) + 1
            
            if (bbox_h, bbox_w) not in interior_dims:
                continue
            
            # Determine rectangle dimensions
            if (bbox_h, bbox_w) == (h - 2, w - 2):
                rh, rw = h, w
            else:
                rh, rw = w, h
            
            sr = min(comp_rs) - 1
            sc = min(comp_cs) - 1
            
            # Check bounds
            if sr < 0 or sr + rh > rows or sc < 0 or sc + rw > cols:
                continue
            
            # Check border is all bg
            border_ok = True
            for dc in range(rw):
                if grid[sr][sc + dc] != bg or grid[sr + rh - 1][sc + dc] != bg:
                    border_ok = False
                    break
            if border_ok:
                for dr in range(rh):
                    if grid[sr + dr][sc] != bg or grid[sr + dr][sc + rw - 1] != bg:
                        border_ok = False
                        break
            
            # Check doesn't overlap with template
            if border_ok:
                if not (sr > r2 or sr + rh - 1 < r1 or sc > c2 or sc + rw - 1 < c1):
                    border_ok = False
            
            if border_ok:
                valid_placements.append((sr, sc, rh, rw, comp, is_independent(comp)))
        
        # Filter: prefer independent-set components; if none, use all
        indep_placements = [p for p in valid_placements if p[5]]
        if indep_placements:
            chosen = indep_placements
        else:
            chosen = valid_placements
        
        # Stamp chosen placements
        for sr, sc, rh, rw, comp, _ in chosen:
            for dr in range(rh):
                for dc in range(rw):
                    r, c = sr + dr, sc + dc
                    if output[r][c] == bg:
                        output[r][c] = frame_color
    
    return output
