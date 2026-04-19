from collections import Counter, deque

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    half_w = W // 2
    half_h = H // 2
    
    def get_info(cells_gen, total):
        freq = Counter(cells_gen)
        bg = freq.most_common(1)[0][0]
        return bg, total - freq[bg]
    
    v_l_bg, v_l_n = get_info((grid[r][c] for r in range(H) for c in range(half_w)), H*half_w)
    v_r_bg, v_r_n = get_info((grid[r][c] for r in range(H) for c in range(half_w, W)), H*(W-half_w))
    h_t_bg, h_t_n = get_info((grid[r][c] for r in range(half_h) for c in range(W)), half_h*W)
    h_b_bg, h_b_n = get_info((grid[r][c] for r in range(half_h, H) for c in range(W)), (H-half_h)*W)
    
    v_ratio = max(v_l_n, v_r_n) / max(1, min(v_l_n, v_r_n))
    h_ratio = max(h_t_n, h_b_n) / max(1, min(h_t_n, h_b_n))
    
    if v_ratio >= h_ratio:
        if v_l_n > v_r_n:
            return _solve(grid, H, W, 'v', half_w, 'left', v_l_bg, v_r_bg)
        else:
            return _solve(grid, H, W, 'v', half_w, 'right', v_r_bg, v_l_bg)
    else:
        if h_t_n > h_b_n:
            return _solve(grid, H, W, 'h', half_h, 'top', h_t_bg, h_b_bg)
        else:
            return _solve(grid, H, W, 'h', half_h, 'bot', h_b_bg, h_t_bg)

def _solve(grid, H, W, split, half, shapes_side, s_bg, c_bg):
    if split == 'v':
        sH, sW = H, half
        cH, cW = H, W - half
        if shapes_side == 'left':
            sg = lambda r,c: grid[r][c]
            cg = lambda r,c: grid[r][c+half]
        else:
            sg = lambda r,c: grid[r][c+half]
            cg = lambda r,c: grid[r][c]
    else:
        sH, sW = half, W
        cH, cW = H - half, W
        if shapes_side == 'top':
            sg = lambda r,c: grid[r][c]
            cg = lambda r,c: grid[r+half][c]
        else:
            sg = lambda r,c: grid[r+half][c]
            cg = lambda r,c: grid[r][c]
    
    vis = [[False]*sW for _ in range(sH)]
    shapes = []
    for r in range(sH):
        for c in range(sW):
            if not vis[r][c] and sg(r,c) != s_bg:
                q = deque([(r,c)])
                vis[r][c] = True
                cells = []
                while q:
                    cr,cc = q.popleft()
                    cells.append((cr,cc,sg(cr,cc)))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc = cr+dr,cc+dc
                        if 0<=nr<sH and 0<=nc<sW and not vis[nr][nc] and sg(nr,nc)!=s_bg:
                            vis[nr][nc]=True
                            q.append((nr,nc))
                shapes.append(cells)
    
    clue_set = {}
    for r in range(cH):
        for c in range(cW):
            v = cg(r,c)
            if v != c_bg:
                clue_set[(r,c)] = v
    
    candidates = []
    for si, shape in enumerate(shapes):
        cc = Counter(v for _,_,v in shape)
        if len(cc) < 2:
            continue
        dom = cc.most_common(1)[0][0]
        minority = [(r,c,v) for r,c,v in shape if v != dom]
        min_size = len(minority)
        
        best_off = None
        best_match = 0
        tested = set()
        for mr,mc,mv in minority:
            for (cr,cc_),cv in clue_set.items():
                if cv == mv:
                    dr,dc = cr-mr, cc_-mc
                    if (dr,dc) in tested:
                        continue
                    tested.add((dr,dc))
                    match = sum(1 for r2,c2,v2 in minority 
                               if (r2+dr,c2+dc) in clue_set and clue_set[(r2+dr,c2+dc)]==v2)
                    if match > best_match:
                        best_match = match
                        best_off = (dr,dc)
        
        if best_off and best_match >= 1:
            frac = best_match / min_size
            candidates.append((frac, best_match, min_size, si, shape, best_off))
    
    # Sort by fraction desc, then match desc
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    
    used = set()
    placements = []
    for frac, _, _, si, shape, (dr,dc) in candidates:
        cc = Counter(v for _,_,v in shape)
        dom = cc.most_common(1)[0][0]
        minority = [(r,c,v) for r,c,v in shape if v != dom]
        
        matched = [(r+dr,c+dc) for r,c,v in minority 
                   if (r+dr,c+dc) in clue_set and clue_set[(r+dr,c+dc)]==v and (r+dr,c+dc) not in used]
        if len(matched) == 0:
            continue
        
        for pos in matched:
            used.add(pos)
        placements.append((shape, dr, dc))
    
    # Single-color shapes with s_bg-colored clues
    bg_clues = [(r,c) for (r,c),v in clue_set.items() if v == s_bg and (r,c) not in used]
    if bg_clues:
        for shape in shapes:
            colors = set(v for _,_,v in shape)
            if len(colors) > 1:
                continue
            shape_set = set((r,c) for r,c,_ in shape)
            min_r = min(r for r,c in shape_set)
            max_r = max(r for r,c in shape_set)
            min_c = min(c for r,c in shape_set)
            max_c = max(c for r,c in shape_set)
            bbox_bg = [(r,c) for r in range(min_r,max_r+1) for c in range(min_c,max_c+1) if (r,c) not in shape_set]
            
            best_off = None
            best_match = 0
            tested = set()
            bg_set = set(bg_clues)
            for br,bc in bbox_bg:
                for cr,cc_ in bg_clues:
                    dr,dc = cr-br, cc_-bc
                    if (dr,dc) in tested:
                        continue
                    tested.add((dr,dc))
                    match = sum(1 for r2,c2 in bbox_bg if (r2+dr,c2+dc) in bg_set)
                    if match > best_match:
                        best_match = match
                        best_off = (dr,dc)
            
            if best_off and best_match >= 2:
                placements.append((shape, best_off[0], best_off[1]))

    # Handle shapes whose minority color == canvas bg (transparent minority).
    # These cannot be matched against canvas clues (all canvas bg cells look the same).
    # Rule: if a transparent-minority shape's column range in shape-space touches
    # the primary placed shape's column range (sharing exactly one boundary column),
    # place it to the right of the primary shape in canvas.
    if placements:
        primary_shape, primary_dr, primary_dc = placements[0]
        pcells = [(r + primary_dr, c + primary_dc)
                  for r, c, v in primary_shape
                  if 0 <= r + primary_dr < cH and 0 <= c + primary_dc < cW]
        if pcells:
            p_top = min(r for r, c in pcells)
            p_bot = max(r for r, c in pcells)
            p_right = max(c for r, c in pcells)
            p_ss_left = min(c for r, c, v in primary_shape)
            p_ss_right = max(c for r, c, v in primary_shape)

            cbg_touching = []
            for shape in shapes:
                cc2 = Counter(v for _, _, v in shape)
                dom2 = cc2.most_common(1)[0][0]
                min_vals2 = set(v for _, _, v in shape if v != dom2)
                if min_vals2 != {c_bg}:
                    continue
                ss_cols = [c for _, c, _ in shape]
                ss_left_s = min(ss_cols)
                ss_right_s = max(ss_cols)
                if ss_right_s == p_ss_left:
                    cbg_touching.append((p_ss_left, ss_left_s, shape))
                elif ss_left_s == p_ss_right:
                    cbg_touching.append((p_ss_right, ss_left_s, shape))

            cbg_touching.sort(key=lambda x: (x[0], x[1]))

            frontier = p_right
            for i, (connect_col, _, shape) in enumerate(cbg_touching):
                ss_rows = [r for r, _, _ in shape]
                ss_cols = [c for _, c, _ in shape]
                ss_left_s = min(ss_cols)
                ss_top_s = min(ss_rows)
                height = max(ss_rows) - ss_top_s + 1
                width = max(ss_cols) - ss_left_s + 1

                # First shape uses +1 gap (would share boundary col with primary),
                # subsequent shapes share the boundary column (different rows, no conflict).
                col_start = frontier + (1 if i == 0 else 0)
                frontier = col_start + width - 1
                dc = col_start - ss_left_s

                if connect_col == p_ss_left:
                    # Left-touching: place above primary (complementary row position)
                    canvas_bot = cH - 1 - p_bot
                    canvas_top_r = canvas_bot - height + 1
                else:
                    # Right-touching: place starting at same top row as primary
                    canvas_top_r = p_top

                if canvas_top_r < 0 or canvas_top_r + height - 1 >= cH:
                    continue
                if col_start < 0 or frontier >= cW:
                    continue

                dr = canvas_top_r - ss_top_s
                placements.append((shape, dr, dc))

    output = [[c_bg]*cW for _ in range(cH)]
    for shape, dr, dc in placements:
        for sr,sc,sv in shape:
            nr,nc = sr+dr, sc+dc
            if 0<=nr<cH and 0<=nc<cW:
                output[nr][nc] = sv
    
    for (r,c),v in clue_set.items():
        if v == s_bg and 0<=r<cH and 0<=c<cW:
            output[r][c] = v
    
    return output
