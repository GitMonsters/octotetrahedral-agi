def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    from collections import Counter
    from itertools import combinations
    
    bg = Counter(v for r in input_grid for v in r).most_common(1)[0][0]
    
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                non_bg[(r, c)] = input_grid[r][c]
    
    # Find frame: rectangle of corner markers with mostly-bg interior
    corner_candidates = {}
    for (r, c), v in non_bg.items():
        corner_candidates.setdefault(v, []).append((r, c))
    
    frame = None
    best_score = -1
    
    for color, cells in corner_candidates.items():
        if len(cells) < 4:
            continue
        for combo in combinations(cells, 4):
            rs = sorted(set(r for r, c in combo))
            cs = sorted(set(c for r, c in combo))
            if len(rs) != 2 or len(cs) != 2:
                continue
            expected = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
            if expected != set(combo):
                continue
            
            r0, r1 = rs
            c0, c1 = cs
            
            # Interior should be mostly bg
            interior_cells = 0
            interior_bg = 0
            for r in range(r0+1, r1):
                for c in range(c0+1, c1):
                    interior_cells += 1
                    if input_grid[r][c] == bg:
                        interior_bg += 1
            
            if interior_cells == 0:
                continue
            bg_ratio = interior_bg / interior_cells
            if bg_ratio < 0.8:
                continue
            
            # Should have at least one colored edge
            left_vals = [non_bg.get((r, c0)) for r in range(r0+1, r1)]
            right_vals = [non_bg.get((r, c1)) for r in range(r0+1, r1)]
            top_vals = [non_bg.get((r0, c)) for c in range(c0+1, c1)]
            bottom_vals = [non_bg.get((r1, c)) for c in range(c0+1, c1)]
            
            edge_count = sum(1 for v in left_vals + right_vals + top_vals + bottom_vals if v is not None)
            
            # Score: bigger frames with more edges are better
            score = (r1 - r0) * (c1 - c0) + edge_count * 10
            
            if score > best_score:
                best_score = score
                lc = next((v for v in left_vals if v is not None and v != color), None)
                rc = next((v for v in right_vals if v is not None and v != color), None)
                tc = next((v for v in top_vals if v is not None and v != color), None)
                bc = next((v for v in bottom_vals if v is not None and v != color), None)
                frame = {
                    'corner_color': color,
                    'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1,
                    'left_color': lc, 'right_color': rc,
                    'top_color': tc, 'bottom_color': bc,
                }
    
    if frame is None:
        return input_grid
    
    r0, r1, c0, c1 = frame['r0'], frame['r1'], frame['c0'], frame['c1']
    
    # Frame cells
    frame_cells = set()
    for r in range(r0, r1+1):
        for c in [c0, c1]:
            if (r, c) in non_bg:
                frame_cells.add((r, c))
    for c in range(c0, c1+1):
        for r in [r0, r1]:
            if (r, c) in non_bg:
                frame_cells.add((r, c))
    
    # Pattern
    pattern_cells = {k: v for k, v in non_bg.items() if k not in frame_cells}
    if not pattern_cells:
        return input_grid
    
    pr0 = min(r for r, c in pattern_cells)
    pr1 = max(r for r, c in pattern_cells)
    pc0 = min(c for r, c in pattern_cells)
    pc1 = max(c for r, c in pattern_cells)
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    
    pat = [[bg]*pw for _ in range(ph)]
    for (r, c), v in pattern_cells.items():
        pat[r - pr0][c - pc0] = v
    
    # Determine flips
    pattern_colors = set(v for v in pattern_cells.values()) - {bg}
    n_colors = len(pattern_colors)
    
    h_flip = False
    if frame['left_color'] is not None or frame['right_color'] is not None:
        if n_colors >= 2:
            color_avg_col = {}
            for (r, c), v in pattern_cells.items():
                color_avg_col.setdefault(v, []).append(c)
            color_avg_col = {v: sum(cs)/len(cs) for v, cs in color_avg_col.items()}
            
            if frame['left_color'] is not None:
                pattern_left = min(color_avg_col, key=color_avg_col.get)
                if pattern_left != frame['left_color']:
                    h_flip = True
            elif frame['right_color'] is not None:
                pattern_right = max(color_avg_col, key=color_avg_col.get)
                if pattern_right != frame['right_color']:
                    h_flip = True
        else:
            h_flip = True  # single color → always flip
    
    v_flip = False
    if frame['top_color'] is not None or frame['bottom_color'] is not None:
        if n_colors >= 2:
            color_avg_row = {}
            for (r, c), v in pattern_cells.items():
                color_avg_row.setdefault(v, []).append(r)
            color_avg_row = {v: sum(rs)/len(rs) for v, rs in color_avg_row.items()}
            
            if frame['top_color'] is not None:
                pattern_top = min(color_avg_row, key=color_avg_row.get)
                if pattern_top != frame['top_color']:
                    v_flip = True
            elif frame['bottom_color'] is not None:
                pattern_bottom = max(color_avg_row, key=color_avg_row.get)
                if pattern_bottom != frame['bottom_color']:
                    v_flip = True
        else:
            v_flip = True
    
    if h_flip:
        pat = [row[::-1] for row in pat]
    if v_flip:
        pat = pat[::-1]
    
    # Build output
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    out_h = r1 - r0 + 1
    out_w = c1 - c0 + 1
    
    output = [[bg]*out_w for _ in range(out_h)]
    
    # Corners
    cc = frame['corner_color']
    output[0][0] = cc
    output[0][out_w-1] = cc
    output[out_h-1][0] = cc
    output[out_h-1][out_w-1] = cc
    
    # Edges
    for r in range(1, out_h-1):
        if frame['left_color'] is not None: output[r][0] = frame['left_color']
        if frame['right_color'] is not None: output[r][out_w-1] = frame['right_color']
    for c in range(1, out_w-1):
        if frame['top_color'] is not None: output[0][c] = frame['top_color']
        if frame['bottom_color'] is not None: output[out_h-1][c] = frame['bottom_color']
    
    # Place pattern in interior
    for r in range(min(ph, inner_h)):
        for c in range(min(pw, inner_w)):
            output[r + 1][c + 1] = pat[r][c]
    
    return output
