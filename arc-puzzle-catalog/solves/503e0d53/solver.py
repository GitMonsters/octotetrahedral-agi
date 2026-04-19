from collections import Counter, deque
import copy

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    non_bg = set((r,c) for r in range(H) for c in range(W) if grid[r][c] != bg)
    if not non_bg:
        return [row[:] for row in grid]
    
    # Find main block (4-connected, largest component)
    visited = set()
    components = []
    for start in sorted(non_bg):
        if start in visited: continue
        comp, stack = [], [start]
        while stack:
            cell = stack.pop()
            if cell in visited or cell not in non_bg: continue
            visited.add(cell)
            comp.append(cell)
            r,c = cell
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr,c+dc))
        components.append(comp)
    
    largest = max(components, key=len)
    rows_l = [r for r,c in largest]
    cols_l = [c for r,c in largest]
    br1,br2,bc1,bc2 = min(rows_l),max(rows_l),min(cols_l),max(cols_l)
    bh, bw = br2-br1+1, bc2-bc1+1
    template = [[grid[r][c] for c in range(bc1,bc2+1)] for r in range(br1,br2+1)]
    
    def rot90cw(g):
        gh, gw = len(g), len(g[0])
        return [[g[gh-1-cc][rr] for cc in range(gh)] for rr in range(gw)]
    def fliph(g):
        return [list(reversed(row)) for row in g]
    
    def get_unique_transforms(t):
        ts = [t]
        cur = t
        for _ in range(3):
            cur = rot90cw(cur)
            ts.append(cur)
        cur = fliph(t)
        ts.append(cur)
        for _ in range(3):
            cur = rot90cw(cur)
            ts.append(cur)
        seen = []
        unique = []
        for t2 in ts:
            key = tuple(tuple(row) for row in t2)
            if key not in seen:
                seen.append(key)
                unique.append(t2)
        return unique
    
    def place_copy(out, tr, sr, sc):
        th, tw = len(tr), len(tr[0])
        for dr in range(th):
            for dc in range(tw):
                if tr[dr][dc] != bg:
                    nr, nc = sr+dr, sc+dc
                    if 0<=nr<H and 0<=nc<W:
                        out[nr][nc] = tr[dr][dc]
    
    def no_col_overlap(sc, sw):
        return (sc + sw - 1 < bc1) or (sc > bc2)
    
    # Identify marker color
    template_colors = Counter(v for row in template for v in row if v != bg)
    dominant = template_colors.most_common(1)[0][0]
    marker_color = None
    for color, _ in template_colors.most_common():
        if color != dominant:
            marker_color = color
            break
    
    block_bbox = set((r,c) for r in range(br1,br2+1) for c in range(bc1,bc2+1))
    sparse_cells = [(r,c) for r,c in non_bg if (r,c) not in block_bbox]
    
    out = [row[:] for row in grid]
    transforms = get_unique_transforms(template)
    placed = set()
    
    if marker_color is not None:
        # PATTERNED BLOCK: marker matching
        sparse_markers = set(pos for pos in sparse_cells if grid[pos[0]][pos[1]] == marker_color)
        
        for trans in transforms:
            th, tw = len(trans), len(trans[0])
            mp = [(dr,dc) for dr in range(th) for dc in range(tw) if trans[dr][dc]==marker_color]
            if not mp: continue
            for r0,c0 in list(sparse_markers):
                for dr0,dc0 in mp:
                    T, L = r0-dr0, c0-dc0
                    if T<0 or L<0 or T+th>H or L+tw>W: continue
                    key = (T,L,th,tw)
                    if key in placed: continue
                    if all((T+mdr,L+mdc) in sparse_markers for mdr,mdc in mp):
                        if no_col_overlap(L, tw):
                            place_copy(out, trans, T, L)
                            placed.add(key)
    elif not sparse_cells:
        # NO SPARSE PIXELS: diagonal walk + left-wall bounce + right-edge wrap
        def flipv(g):
            return list(reversed(g))

        # Walk diagonally UP-LEFT from template placing original-orientation copies
        r, c = br1, bc1
        parity = 0
        off_r, off_c = None, None  # position that went off left edge
        while True:
            dr = -(bh + bw) if parity == 0 else (bh - bw)
            dc = -bh
            r_new, c_new = r + dr, c + dc
            if c_new < 0:
                off_r, off_c = r_new, c_new
                break
            if r_new < 0 or r_new + bh > H:
                break
            place_copy(out, template, r_new, c_new)
            r, c = r_new, c_new
            parity = 1 - parity

        # Vertical bouncing at the leftmost column (col 0)
        if c == 0 and off_r is not None:
            bounce_orients = [fliph(template), fliph(flipv(template)),
                              flipv(template), template]
            r_bounce = r
            vert_parity = 0
            orient_idx = 0
            while True:
                v_step = (bh - 1) if vert_parity == 0 else (bh - bw)
                r_bounce += v_step
                if r_bounce + bh > H:
                    break
                place_copy(out, bounce_orients[orient_idx % 4], r_bounce, 0)
                vert_parity = 1 - vert_parity
                orient_idx += 1

        # 90°CW wrap to right edge
        if off_r is not None and off_c is not None:
            cw = rot90cw(template)
            cw_h, cw_w = len(cw), len(cw[0])
            wrap_c = off_c % W
            wrap_r = max(0, off_r)
            if wrap_r + cw_h <= H and wrap_c + cw_w <= W:
                place_copy(out, cw, wrap_r, wrap_c)
    else:
        # SOLID BLOCK: cluster center approach
        sparse_set = set(sparse_cells)
        visited2 = set()
        clusters = []
        for start in sorted(sparse_set):
            if start in visited2: continue
            comp, q = [], deque([start])
            visited2.add(start)
            while q:
                r,c = q.popleft()
                comp.append((r,c))
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr,nc = r+dr,c+dc
                    if (nr,nc) in sparse_set and (nr,nc) not in visited2:
                        visited2.add((nr,nc))
                        q.append((nr,nc))
            clusters.append(comp)
        
        threshold = max(bh, bw)
        min_size = min(bh, bw)
        min_row_filter = br1 - bh // 2
        
        def cheb_dist(cells):
            return min(max(max(0, br1-r, r-br2), max(0, bc1-c, c-bc2)) for r,c in cells)
        
        for cluster in clusters:
            if len(cluster) < min_size:
                continue
            dist = cheb_dist(cluster)
            if dist > threshold:
                continue
            min_r = min(r for r,c in cluster)
            if min_r < min_row_filter:
                continue
            
            rows_c = [r for r,c in cluster]
            cols_c = [c for r,c in cluster]
            cr = (min(rows_c) + max(rows_c)) / 2.0
            cc = (min(cols_c) + max(cols_c)) / 2.0
            cspan_r = max(rows_c) - min(rows_c) + 1
            cspan_c = max(cols_c) - min(cols_c) + 1
            
            # Choose orientation: match aspect ratio
            # For bh x bw block: taller = max(bh,bw) x min(bh,bw)
            if bh == bw:
                chosen_th, chosen_tw = bh, bw
            elif cspan_r >= cspan_c:
                chosen_th, chosen_tw = max(bh,bw), min(bh,bw)
            else:
                chosen_th, chosen_tw = min(bh,bw), max(bh,bw)
            
            T = round(cr - (chosen_th-1)/2.0)
            L = round(cc - (chosen_tw-1)/2.0)
            
            if T<0 or L<0 or T+chosen_th>H or L+chosen_tw>W:
                continue
            if not no_col_overlap(L, chosen_tw):
                continue
            
            key = (T,L,chosen_th,chosen_tw)
            if key in placed:
                continue
            
            # Find the transform with matching dimensions
            for trans in transforms:
                if len(trans)==chosen_th and len(trans[0])==chosen_tw:
                    place_copy(out, trans, T, L)
                    placed.add(key)
                    break
    
    return out
