from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(H) for c in range(W)]
    cc = Counter(flat)
    bg = cc.most_common(1)[0][0]
    non_bg = {c: n for c, n in cc.items() if c != bg}
    
    if not non_bg:
        return [row[:] for row in grid]
    
    result = [row[:] for row in grid]
    shape = max(non_bg, key=non_bg.get)
    marker_colors = sorted([c for c in non_bg if c != shape])
    
    # Find ALL non-bg connected components (4-connected, including markers)
    visited = set()
    all_comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or (r, c) in visited:
                continue
            stack = [(r, c)]
            comp = []
            while stack:
                cr, cc2 = stack.pop()
                if (cr, cc2) in visited or cr < 0 or cr >= H or cc2 < 0 or cc2 >= W:
                    continue
                if grid[cr][cc2] == bg:
                    continue
                visited.add((cr, cc2))
                comp.append((cr, cc2, grid[cr][cc2]))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cr + dr, cc2 + dc))
            all_comps.append(comp)
    
    # Classify components
    big_comps = []  # 5+ cell components with both shape and marker
    singletons = []  # single shape cells
    small_lshapes = []  # 3-cell L-shapes (pure shape)
    
    for comp in all_comps:
        colors = set(v for _, _, v in comp)
        shape_cells = [(r, c) for r, c, v in comp if v == shape]
        marker_cells = [(r, c, v) for r, c, v in comp if v != shape]
        
        if len(comp) == 1 and comp[0][2] == shape:
            singletons.append(comp[0])
        elif len(shape_cells) == 4 and len(marker_cells) >= 1:
            big_comps.append(comp)
        elif len(comp) == 3 and len(colors) == 1 and shape in colors:
            r1 = min(r for r, c, _ in comp)
            c1 = min(c for r, c, _ in comp)
            r2 = max(r for r, c, _ in comp)
            c2 = max(c for r, c, _ in comp)
            if r2 - r1 == 1 and c2 - c1 == 1:
                comp_set = set((r, c) for r, c, _ in comp)
                corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                missing = [p for p in corners if p not in comp_set]
                if len(missing) == 1:
                    small_lshapes.append((comp, missing[0]))
    
    # CASE 1: No small L-shapes and no big comps -> return as-is
    if not small_lshapes and not big_comps:
        return result
    
    # CASE 2: Small L-shapes only (like train 2)
    if small_lshapes and not big_comps:
        mc = marker_colors[0] if marker_colors else 5
        for comp, missing in small_lshapes:
            mr, mc2 = missing
            comp_set = set((r, c) for r, c, _ in comp)
            elbow = None
            for r, c, _ in comp:
                if abs(r - mr) == 1 and abs(c - mc2) == 1:
                    elbow = (r, c)
                    break
            if not elbow:
                continue
            arms = [(r - elbow[0], c - elbow[1]) for r, c, _ in comp if (r, c) != elbow]
            bg_cands = []
            for dr, dc in arms:
                nr, nc = mr + dr, mc2 + dc
                if 0 <= nr < H and 0 <= nc < W and result[nr][nc] == bg:
                    bg_cands.append((nr, nc, dr, dc))
            
            if len(bg_cands) == 1:
                result[bg_cands[0][0]][bg_cands[0][1]] = mc
            elif len(bg_cands) >= 2:
                # Heuristic: pick candidate not adjacent to any shape cell
                best = None
                for nr, nc, dr, dc in bg_cands:
                    adj_shape = 0
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr2, nc2 = nr + dr2, nc + dc2
                        if 0 <= nr2 < H and 0 <= nc2 < W and grid[nr2][nc2] == shape:
                            adj_shape += 1
                    if best is None or adj_shape < best[1]:
                        best = ((nr, nc), adj_shape)
                if best:
                    result[best[0][0]][best[0][1]] = mc
        return result
    
    # CASE 3: Big components + singletons (like train 0)
    if big_comps:
        # Determine marker colors
        fill_color = None  # color 5 (the one in existing markers)
        border_color = None  # color 3 (the one added to singletons)
        
        for comp in big_comps:
            for r, c, v in comp:
                if v != shape and v != bg:
                    fill_color = v
                    break
            if fill_color:
                break
        
        # Border color is the other marker color
        if len(marker_colors) >= 2:
            border_color = [c for c in marker_colors if c != fill_color][0]
        elif len(marker_colors) == 1:
            border_color = marker_colors[0]
        
        if not fill_color:
            fill_color = 5
        if not border_color:
            border_color = 3
        
        for comp in big_comps:
            r1 = min(r for r, c, _ in comp)
            c1 = min(c for r, c, _ in comp)
            r2 = max(r for r, c, _ in comp)
            c2 = max(c for r, c, _ in comp)
            bH, bW = r2 - r1 + 1, c2 - c1 + 1
            
            # Find existing fill marker position (relative to bbox)
            fill_pos = None
            for r, c, v in comp:
                if v == fill_color:
                    fill_pos = (r - r1, c - c1)
                    break
            
            if not fill_pos:
                continue
            
            # Find bbox corner diagonally adjacent to fill marker
            fr, fc = fill_pos
            corners = [(0, 0), (0, bW - 1), (bH - 1, 0), (bH - 1, bW - 1)]
            for cr, cc2 in corners:
                if abs(cr - fr) == 1 and abs(cc2 - fc) == 1:
                    # This corner should get the new fill marker
                    nr, nc = r1 + cr, c1 + cc2
                    if 0 <= nr < H and 0 <= nc < W and result[nr][nc] == bg:
                        result[nr][nc] = fill_color
                    break
        
        # Handle singletons: add border_color at adjacent position
        for sr, sc, sv in singletons:
            # Find the nearest big comp
            best_comp = None
            best_dist = float('inf')
            for comp in big_comps:
                for r, c, v in comp:
                    d = abs(r - sr) + abs(c - sc)
                    if d < best_dist:
                        best_dist = d
                        best_comp = comp
            
            if not best_comp:
                continue
            
            # Determine bbox orientation of nearest comp
            br1 = min(r for r, c, _ in best_comp)
            bc1 = min(c for r, c, _ in best_comp)
            br2 = max(r for r, c, _ in best_comp)
            bc2 = max(c for r, c, _ in best_comp)
            bH, bW = br2 - br1 + 1, bc2 - bc1 + 1
            
            # Find existing fill pos and added fill direction
            fill_pos = None
            for r, c, v in best_comp:
                if v == fill_color:
                    fill_pos = (r, c)
                    break
            
            if not fill_pos:
                continue
            
            # Find added fill position (bbox corner diag to fill)
            fr, fc = fill_pos[0] - br1, fill_pos[1] - bc1
            corners = [(0, 0), (0, bW - 1), (bH - 1, 0), (bH - 1, bW - 1)]
            added_fill = None
            for cr, cc2 in corners:
                if abs(cr - fr) == 1 and abs(cc2 - fc) == 1:
                    added_fill = (br1 + cr, bc1 + cc2)
                    break
            
            if not added_fill:
                continue
            
            # Fill direction: from existing fill to added fill
            fill_dr = added_fill[0] - fill_pos[0]
            fill_dc = added_fill[1] - fill_pos[1]
            
            # For vertical (tall) bbox: border goes in row direction
            # For horizontal (wide) bbox: border goes in col direction
            if bH > bW:  # vertical
                border_dr = fill_dr
                border_dc = 0
            else:  # horizontal
                border_dr = 0
                border_dc = fill_dc
            
            nr, nc = sr + border_dr, sc + border_dc
            if 0 <= nr < H and 0 <= nc < W and result[nr][nc] == bg:
                result[nr][nc] = border_color
    
    return result
