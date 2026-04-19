from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    out = [row[:] for row in grid]
    
    visited = set()
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or (r,c) in visited:
                continue
            stack = [(r,c)]
            comp = []
            while stack:
                cr,cc = stack.pop()
                if (cr,cc) in visited or not(0<=cr<H and 0<=cc<W):
                    continue
                if grid[cr][cc] == bg:
                    continue
                visited.add((cr,cc))
                comp.append((cr,cc,grid[cr][cc]))
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr==0 and dc==0: continue
                        stack.append((cr+dr,cc+dc))
            if comp:
                comps.append(comp)
    
    def get_4conn_subgroups(positions):
        pos_set = set(positions)
        vis = set()
        groups = []
        for p in positions:
            if p in vis:
                continue
            group = []
            stk = [p]
            while stk:
                cp = stk.pop()
                if cp in vis: continue
                vis.add(cp)
                group.append(cp)
                cr, cc = cp
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    np = (cr+dr, cc+dc)
                    if np in pos_set and np not in vis:
                        stk.append(np)
            groups.append(group)
        return groups
    
    for comp in comps:
        colors = Counter(v for _,_,v in comp)
        
        if len(colors) == 1:
            all_pos = [(r,c) for r,c,v in comp]
            subgroups = get_4conn_subgroups(all_pos)
            if len(subgroups) <= 1:
                continue
            subgroups.sort(key=len, reverse=True)
            template_cells = set(subgroups[0])
            the_color = comp[0][2]
            seeds = [(r,c,the_color) for r,c in all_pos if (r,c) not in template_cells]
        else:
            maj_color = colors.most_common(1)[0][0]
            template_cells = set((r,c) for r,c,v in comp if v == maj_color)
            seeds = [(r,c,v) for r,c,v in comp if v != maj_color]
        
        # 2-cell equal-count pairs
        if len(comp) == 2 and len(colors) == 2:
            c1, c2 = sorted(comp)
            r1,c1c,v1 = c1; r2,c2c,v2 = c2
            dr, dc = r2-r1, c2c-c1c
            
            if abs(dr) == 1 and abs(dc) == 1:
                for cr,cc,cv in comp:
                    away_r = -dr if cv==v1 else dr
                    away_c = -dc if cv==v1 else dc
                    nr1, nc1 = cr+away_r, cc+away_c
                    nr2, nc2 = nr1+away_r, nc1-away_c
                    for nr,nc in [(nr1,nc1),(nr2,nc2)]:
                        if 0<=nr<H and 0<=nc<W and out[nr][nc]==bg:
                            out[nr][nc] = cv
            elif dr == 0:
                ext_dir = 1 if r1 < H/2 else -1
                for cr,cc,cv in comp:
                    other_c = c2c if cv==v1 else c1c
                    side = -1 if cc < other_c else 1
                    center_r, center_c = cr+ext_dir, cc+side
                    for bdr in [-1,0,1]:
                        for bdc in [-1,0,1]:
                            br, bc = center_r+bdr, center_c+bdc
                            if 0<=br<H and 0<=bc<W:
                                if br == cr+ext_dir and bc == cc:
                                    continue
                                if out[br][bc] == bg:
                                    out[br][bc] = cv
            elif dc == 0:
                ext_dir = 1 if c1c < W/2 else -1
                for cr,cc,cv in comp:
                    other_r = r2 if cv==v1 else r1
                    side = -1 if cr < other_r else 1
                    center_r, center_c = cr+side, cc+ext_dir
                    for bdr in [-1,0,1]:
                        for bdc in [-1,0,1]:
                            br, bc = center_r+bdr, center_c+bdc
                            if 0<=br<H and 0<=bc<W:
                                if br == cr and bc == cc+ext_dir:
                                    continue
                                if out[br][bc] == bg:
                                    out[br][bc] = cv
            continue
        
        # Chain reflection
        all_shapes = {}
        shapes_list = [template_cells.copy()]
        for pos in template_cells:
            all_shapes[pos] = (grid[pos[0]][pos[1]], 0)
        
        remaining = list(seeds)
        processed = True
        while processed and remaining:
            processed = False
            for i, (sr,sc,sv) in enumerate(remaining):
                best_contact = None
                best_priority = None
                
                for ddr, ddc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    nr, nc = sr+ddr, sc+ddc
                    if (nr,nc) in all_shapes:
                        _, gid = all_shapes[(nr,nc)]
                        is_4adj = abs(ddr)+abs(ddc) == 1
                        priority = (1 if is_4adj else 0, gid)
                        if best_contact is None or priority > best_priority:
                            best_contact = (nr, nc, ddr, ddc)
                            best_priority = priority
                
                if best_contact is None:
                    continue
                
                nr, nc, ddr, ddc = best_contact
                _, gid = all_shapes[(nr, nc)]
                source = shapes_list[gid]
                contact = (nr, nc)
                dr_dir, dc_dir = -ddr, -ddc
                
                new_shape = set()
                for (cr,cc) in source:
                    rel_r, rel_c = cr - contact[0], cc - contact[1]
                    if dc_dir != 0 and dr_dir == 0:
                        ref_r, ref_c = rel_r, -rel_c
                    elif dr_dir != 0 and dc_dir == 0:
                        ref_r, ref_c = -rel_r, rel_c
                    else:
                        ref_r, ref_c = -rel_r, -rel_c
                    ar, ac = sr + ref_r, sc + ref_c
                    if 0 <= ar < H and 0 <= ac < W:
                        new_shape.add((ar, ac))
                
                # Special case: V-flip of horizontal line with centered seed
                if dr_dir != 0 and dc_dir == 0:
                    src_rows = set(r for r,c in source)
                    if len(src_rows) == 1:
                        src_row = list(src_rows)[0]
                        src_cols = sorted(c for r,c in source)
                        n = len(src_cols)
                        if n % 2 == 1:
                            mid = src_cols[n // 2]
                            if sc == mid:
                                min_c, max_c = src_cols[0], src_cols[-1]
                                d_left = min_c
                                d_right = W - 1 - max_c
                                if d_right <= d_left:
                                    fold_c = max_c
                                    ext_c = max_c + 1
                                else:
                                    fold_c = min_c
                                    ext_c = min_c - 1
                                fold_pos = (sr, fold_c)
                                new_shape.discard(fold_pos)
                                if 0 <= ext_c < W:
                                    ext_pos = (src_row, ext_c)
                                    tmpl_color = grid[src_row][src_cols[0]]
                                    if out[ext_pos[0]][ext_pos[1]] == bg:
                                        out[ext_pos[0]][ext_pos[1]] = tmpl_color
                
                # Similarly for H-flip of vertical line with centered seed
                if dc_dir != 0 and dr_dir == 0:
                    src_cols = set(c for r,c in source)
                    if len(src_cols) == 1:
                        src_col = list(src_cols)[0]
                        src_rows = sorted(r for r,c in source)
                        n = len(src_rows)
                        if n % 2 == 1:
                            mid = src_rows[n // 2]
                            if sr == mid:
                                min_r, max_r = src_rows[0], src_rows[-1]
                                d_top = min_r
                                d_bottom = H - 1 - max_r
                                if d_bottom <= d_top:
                                    fold_r = max_r
                                    ext_r = max_r + 1
                                else:
                                    fold_r = min_r
                                    ext_r = min_r - 1
                                fold_pos = (fold_r, sc)
                                new_shape.discard(fold_pos)
                                if 0 <= ext_r < H:
                                    ext_pos = (ext_r, src_col)
                                    tmpl_color = grid[src_rows[0]][src_col]
                                    if out[ext_pos[0]][ext_pos[1]] == bg:
                                        out[ext_pos[0]][ext_pos[1]] = tmpl_color
                
                ngid = len(shapes_list)
                shapes_list.append(new_shape)
                
                for pos in new_shape:
                    if pos not in all_shapes:
                        all_shapes[pos] = (sv, ngid)
                    if out[pos[0]][pos[1]] == bg:
                        out[pos[0]][pos[1]] = sv
                
                if (sr,sc) not in all_shapes:
                    all_shapes[(sr,sc)] = (sv, ngid)
                
                remaining.pop(i)
                processed = True
                break
    
    return out


solve = transform  # catalog alias
