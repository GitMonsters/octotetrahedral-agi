from collections import Counter, deque

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    cc = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg = cc.most_common(1)[0][0]

    visited = [[False]*cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                q = deque([(r, c)]); visited[r][c] = True; cells = []
                while q:
                    cr, cc2 = q.popleft(); cells.append((cr, cc2, grid[cr][cc2]))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc2+dc
                        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc] and grid[nr][nc]!=bg:
                            visited[nr][nc] = True; q.append((nr, nc))
                comps.append(cells)

    comps.sort(key=len, reverse=True)

    # Case 1: large shape exists (>= 10 cells) → hole-filling logic
    if comps and len(comps[0]) >= 10:
        big = comps[0]; patches = comps[1:]
        big_colors = Counter(c for _,_,c in big)
        fill = big_colors.most_common(1)[0][0]
        big_rs = [r for r,c,_ in big]; big_cs = [c for r,c,_ in big]
        min_r, max_r = min(big_rs), max(big_rs)
        min_c, max_c = min(big_cs), max(big_cs)
        H = max_r - min_r + 1; W = max_c - min_c + 1

        big_grid = [[bg]*W for _ in range(H)]
        for r, c, v in big: big_grid[r-min_r][c-min_c] = v

        hole_vis = [[False]*W for _ in range(H)]
        hole_groups = []
        for r in range(H):
            for c in range(W):
                if big_grid[r][c] == bg and not hole_vis[r][c]:
                    q = deque([(r,c)]); hole_vis[r][c] = True; hcells = []
                    while q:
                        cr, cc2 = q.popleft(); hcells.append((cr, cc2))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc2+dc
                            if 0<=nr<H and 0<=nc<W and not hole_vis[nr][nc] and big_grid[nr][nc]==bg:
                                hole_vis[nr][nc] = True; q.append((nr, nc))
                    hole_groups.append(hcells)

        def get_bbox(cells_2d):
            rs=[r for r,c in cells_2d]; cs=[c for r,c in cells_2d]
            return min(rs),max(rs),min(cs),max(cs)

        for hole in hole_groups:
            hr1,hr2,hc1,hc2 = get_bbox(hole)
            hH,hW = hr2-hr1+1,hc2-hc1+1
            hole_set = set(hole)
            # Find patch that fits this hole
            for patch in patches:
                p_rs=[r for r,c,_ in patch]; p_cs=[c for r,c,_ in patch]
                pr1,pr2,pc1,pc2 = min(p_rs),max(p_rs),min(p_cs),max(p_cs)
                pH,pW = pr2-pr1+1,pc2-pc1+1
                if pH==hH and pW==hW:
                    # Build 8 orientations of the patch cells
                    def orientations(cells, h, w):
                        result = []
                        for flip in [False, True]:
                            for rot in range(4):
                                transformed = []
                                for r2, c2, v2 in cells:
                                    dr2, dc2 = r2 - pr1, c2 - pc1
                                    if flip:
                                        dc2 = w - 1 - dc2
                                    for _ in range(rot):
                                        dr2, dc2 = dc2, h - 1 - dr2
                                    transformed.append((dr2, dc2, v2))
                                new_h = h if rot % 2 == 0 else w
                                new_w = w if rot % 2 == 0 else h
                                if new_h == hH and new_w == hW:
                                    result.append(transformed)
                        return result

                    patch_cells = [(r, c, v) for r, c, v in patch]
                    placed = False
                    for oriented in orientations(patch_cells, pH, pW):
                        patch_rel = {(dr2, dc2): v2 for dr2, dc2, v2 in oriented}
                        # Match: patch BG (empty positions in bbox) must NOT land on holes
                        # Fill and special cells can be anywhere
                        match = True
                        for dr2 in range(hH):
                            for dc2 in range(hW):
                                is_hole = (hr1+dr2, hc1+dc2) in hole_set
                                has_cell = (dr2, dc2) in patch_rel
                                cell_val = patch_rel.get((dr2, dc2))
                                # Patch BG at a hole → doesn't cover the hole → fail
                                if not has_cell and is_hole:
                                    match = False; break
                                # Fill cell at non-hole → shape mismatch → fail
                                if has_cell and cell_val == fill and not is_hole:
                                    match = False; break
                            if not match: break
                        if match:
                            # Place patch: cells get their value, BG positions get cleared
                            for dr2 in range(hH):
                                for dc2 in range(hW):
                                    val = patch_rel.get((dr2, dc2))
                                    if val is not None:
                                        big_grid[hr1+dr2][hc1+dc2] = val
                                    else:
                                        big_grid[hr1+dr2][hc1+dc2] = bg
                            placed = True; break
                    if placed: break

        return big_grid

    # Case 2: no large shape → use output size (H-11, W-10)
    oH = rows - 11
    oW = cols - 10
    if oH <= 0 or oW <= 0:
        return [[bg]*cols for _ in range(rows)]

    # Collect all non-bg cells
    all_cells = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols) if grid[r][c] != bg]

    # Find 3-marker (cell with value that's "special" - look for unique/minority color)
    val_counts = Counter(v for r,c,v in all_cells)
    # The marker is the least common non-bg value
    marker_val = None
    shape_val = None
    if len(val_counts) >= 2:
        sorted_vals = sorted(val_counts.items(), key=lambda x: x[1])
        marker_val = sorted_vals[0][0]
        shape_val = sorted_vals[-1][0]
    elif len(val_counts) == 1:
        shape_val = list(val_counts.keys())[0]

    shape_cells = [(r, c) for r, c, v in all_cells if v == shape_val] if shape_val else []
    marker_cells = [(r, c, v) for r, c, v in all_cells if v == marker_val] if marker_val else []

    if not shape_cells:
        return [[bg]*oH for _ in range(oW)]  # fallback

    output = [[bg]*oW for _ in range(oH)]

    if marker_val is not None and marker_cells:
        # Case 2a: has special marker → place shape relative to marker position
        mr, mc, mv = marker_cells[0]
        # Shape bbox
        s_min_r = min(r for r,c in shape_cells)
        s_min_c = min(c for r,c in shape_cells)
        s_max_r = max(r for r,c in shape_cells)
        # Place shape at output (mr - mv, mc - mv//3)
        out_r = mr - mv
        out_c = mc - mv // 3
        for r, c in shape_cells:
            nr = out_r + (r - s_min_r)
            nc = out_c + (c - s_min_c)
            if 0 <= nr < oH and 0 <= nc < oW:
                output[nr][nc] = shape_val
        # Place marker at (shape_end_row + mv, out_c)
        s_max_out_r = out_r + (s_max_r - s_min_r)
        mr_out = s_max_out_r + mv
        if 0 <= mr_out < oH and 0 <= out_c < oW:
            output[mr_out][out_c] = mv
    else:
        # Case 2b: no marker → rotate 90° CCW and place with offset
        s_min_r = min(r for r,c in shape_cells)
        s_max_r = max(r for r,c in shape_cells)
        s_min_c = min(c for r,c in shape_cells)
        s_max_c = max(c for r,c in shape_cells)
        bbox_W = s_max_c - s_min_c + 1

        out_top = s_min_r + s_min_c
        out_left = s_min_r

        for r, c in shape_cells:
            dr = r - s_min_r; dc = c - s_min_c
            # 90° CCW: (dr,dc) → (bbox_W-1-dc, dr)
            new_r = bbox_W - 1 - dc
            new_c = dr
            nr = new_r + out_top
            nc = new_c + out_left
            if 0 <= nr < oH and 0 <= nc < oW:
                output[nr][nc] = shape_val

    return output
