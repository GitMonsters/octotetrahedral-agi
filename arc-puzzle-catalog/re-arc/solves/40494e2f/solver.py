def transform(grid):
    from collections import Counter, deque
    
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    
    cc = Counter(grid[r][c] for r in range(h) for c in range(w))
    bg = cc.most_common(1)[0][0]
    non_bg = {c: n for c, n in cc.items() if c != bg}
    if 2 not in non_bg or len(non_bg) < 2:
        return out
    sprite_color = [c for c in non_bg if c != 2][0]
    
    sp_cells = [(r,c) for r in range(h) for c in range(w) if grid[r][c] == sprite_color]
    if not sp_cells:
        return out
    sr0 = min(r for r,c in sp_cells); sc0 = min(c for r,c in sp_cells)
    sr1 = max(r for r,c in sp_cells); sc1 = max(c for r,c in sp_cells)
    sh, sw = sr1-sr0+1, sc1-sc0+1
    sprite = [[grid[sr0+r][sc0+c] for c in range(sw)] for r in range(sh)]
    
    # Find isolated boundary cells for extension
    ext_cells = []
    left_arm = 0
    for r in range(sh):
        for c in range(sw):
            if sprite[r][c] != sprite_color:
                continue
            if c == 0 and sum(1 for rr2 in range(sh) if sprite[rr2][0] == sprite_color) == 1:
                ext_cells.append((r, c, 'LEFT'))
                # Compute LEFT arm length: distance from crossing to this cell
                # Find the crossing (bg or 2 cell in the same row that separates arms)
                left_arm = c + 1  # cells from col 0 to this cell
                # Actually, LEFT arm = number of cells to extend = position of isolated cell
                # When stamp at pc, the isolated cell is at pc+c = pc+0 = pc
                # Extension goes from pc-1 to 0 = pc cells
            if c == sw-1 and sum(1 for rr2 in range(sh) if sprite[rr2][sw-1] == sprite_color) == 1:
                ext_cells.append((r, c, 'RIGHT'))
            if r == 0 and sum(1 for cc2 in range(sw) if sprite[0][cc2] == sprite_color) == 1:
                ext_cells.append((r, c, 'UP'))
            if r == sh-1 and sum(1 for cc2 in range(sw) if sprite[sh-1][cc2] == sprite_color) == 1:
                ext_cells.append((r, c, 'DOWN'))
    
    has_left = any(d == 'LEFT' for _,_,d in ext_cells)
    has_right = any(d == 'RIGHT' for _,_,d in ext_cells)
    has_up = any(d == 'UP' for _,_,d in ext_cells)
    has_down = any(d == 'DOWN' for _,_,d in ext_cells)
    has_h = has_left or has_right
    has_v = has_up or has_down
    do_tile = has_h and has_v
    
    # Find LEFT arm length (# of sprite-color cells from LEFT boundary to crossing)
    # This is the pc value for single-stamp extension cases
    left_arm_len = 0
    if has_left:
        for er, ec, ed in ext_cells:
            if ed == 'LEFT':
                # Count sprite-color cells in the same row from col 0 up to the crossing
                row = er
                arm = 0
                for c in range(sw):
                    if sprite[row][c] == sprite_color:
                        arm += 1
                    else:
                        break
                left_arm_len = arm
                break
    
    # Find rectangles
    visited = [[False]*w for _ in range(h)]
    for r in range(sr0, sr1+1):
        for c in range(sc0, sc1+1):
            visited[r][c] = True
    rects = []
    non_rect_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2 and not visited[r][c]:
                q = deque([(r,c)]); visited[r][c] = True; cells = set()
                while q:
                    cr, cc2 = q.popleft(); cells.add((cr,cc2))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc2+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==2:
                            visited[nr][nc] = True; q.append((nr,nc))
                min_r = min(rr for rr,_ in cells); max_r = max(rr for rr,_ in cells)
                min_c = min(cc2 for _,cc2 in cells); max_c = max(cc2 for _,cc2 in cells)
                rh, rw = max_r-min_r+1, max_c-min_c+1
                if rh*rw == len(cells) and rh >= sh and rw >= sw:
                    rects.append((min_r, min_c, rh, rw))
                else:
                    non_rect_cells.extend(cells)
    
    def gen_output(rh, rw, pr, pc):
        result = set()
        stamps = [(pr, pc)]
        if do_tile:
            k = 1
            while True:
                nr2 = pr + k*sh; nc2 = pc - k*(sw-1)
                if 0<=nc2 and nc2+sw<=rw and nr2+sh<=rh:
                    stamps.append((nr2, nc2)); k += 1
                else: break
        for spr, spc in stamps:
            for r in range(sh):
                for c in range(sw):
                    if sprite[r][c] == sprite_color:
                        result.add((spr+r, spc+c))
            for er, ec, d in ext_cells:
                ar, ac = spr+er, spc+ec
                if d == 'LEFT':
                    for cc2 in range(ac-1, -1, -1): result.add((ar, cc2))
                elif d == 'RIGHT':
                    for cc2 in range(ac+1, rw): result.add((ar, cc2))
                elif d == 'UP':
                    for rr2 in range(ar-1, -1, -1): result.add((rr2, ac))
                elif d == 'DOWN':
                    for rr2 in range(ar+1, rh): result.add((rr2, ac))
        return result
    
    def find_position(rr, rc, rh, rw):
        avail_r = rh - sh + 1
        avail_c = rw - sw + 1
        
        # ROW
        pr = (sw - rh) % sh
        if pr == 0 and not has_v:
            pr = round((rh - sh) / 2)
        if pr >= avail_r:
            pr = avail_r - 1
        
        # COLUMN
        if do_tile:
            # Tiling case: compute n_tiles
            n_rows = (rh - pr) // sh
            n_cols = (rw - sw) // (sw - 1) + 1 if sw > 1 else 1
            n_tiles = min(n_rows, n_cols)
            pc = max((n_tiles - 1) * (sw - 1), left_arm_len)
            if pc >= avail_c:
                pc = avail_c - 1
        elif has_h and not has_v:
            # H-only extension: use (rw + rh) % avail
            pc = (rw + rh) % avail_c
        else:
            # No extensions: use sprite-origin based formula
            n_v = (rh - pr) // sh
            pc = (sr0 + sc0 - rc + n_v - 1) % avail_c
        
        return pr, pc
    
    for rr, rc, rh, rw in rects:
        pr, pc = find_position(rr, rc, rh, rw)
        cells = gen_output(rh, rw, pr, pc)
        for r_rel, c_rel in cells:
            out[rr + r_rel][rc + c_rel] = sprite_color
    
    # Clear entire sprite bbox area to background
    for r in range(sr0, sr1+1):
        for c in range(sc0, sc1+1):
            out[r][c] = bg
    for r, c in non_rect_cells:
        out[r][c] = bg
    
    return out
