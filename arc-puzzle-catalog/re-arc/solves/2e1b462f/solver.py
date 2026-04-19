from collections import Counter

def solve(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not cells:
        return grid

    vert_lines = []
    for col in range(cols):
        col_cells = sorted([r2 for r2, c2 in cells if c2 == col])
        if len(col_cells) < 3:
            continue
        best_start, best_end = col_cells[0], col_cells[0]
        start = col_cells[0]
        for i in range(1, len(col_cells)):
            if col_cells[i] == col_cells[i-1] + 1:
                if col_cells[i] - start > best_end - best_start:
                    best_start, best_end = start, col_cells[i]
            else:
                start = col_cells[i]
        if best_end - best_start + 1 >= 3:
            vert_lines.append((col, best_start, best_end))

    horiz_lines = []
    for row in range(rows):
        row_cells = sorted([c2 for r2, c2 in cells if r2 == row])
        if len(row_cells) < 3:
            continue
        best_start, best_end = row_cells[0], row_cells[0]
        start = row_cells[0]
        for i in range(1, len(row_cells)):
            if row_cells[i] == row_cells[i-1] + 1:
                if row_cells[i] - start > best_end - best_start:
                    best_start, best_end = start, row_cells[i]
            else:
                start = row_cells[i]
        if best_end - best_start + 1 >= 3:
            horiz_lines.append((row, best_start, best_end))

    best_vert_pair, best_vert_score = None, 0
    for i in range(len(vert_lines)):
        for j in range(i+1, len(vert_lines)):
            c1, s1, e1 = vert_lines[i]
            c2, s2, e2 = vert_lines[j]
            if s1 == s2:
                shared_len = min(e1, e2) - s1 + 1
                if shared_len >= 3 and shared_len > best_vert_score:
                    best_vert_score = shared_len
                    best_vert_pair = (vert_lines[i], vert_lines[j])

    best_horiz_pair, best_horiz_score = None, 0
    for i in range(len(horiz_lines)):
        for j in range(i+1, len(horiz_lines)):
            r1, s1, e1 = horiz_lines[i]
            r2, s2, e2 = horiz_lines[j]
            if s1 == s2:
                shared_len = min(e1, e2) - s1 + 1
                if shared_len >= 3 and shared_len > best_horiz_score:
                    best_horiz_score = shared_len
                    best_horiz_pair = (horiz_lines[i], horiz_lines[j])

    def build_vert(pair):
        (c1, s1, e1), (c2, s2, e2) = pair
        bcl, bcr = min(c1, c2), max(c1, c2)
        brs = max(s1, s2)
        bre = min(e1, e2)
        fset = set()
        for rr in range(brs, bre+1):
            fset.add((rr, bcl)); fset.add((rr, bcr))
        sc = [(rr, cc) for rr, cc in cells if (rr, cc) not in fset]
        fh, fw = bre-brs+1, bcr-bcl+1
        frame = [[bg]*fw for _ in range(fh)]
        for rr, cc in cells:
            if (rr, cc) in fset:
                frame[rr-brs][cc-bcl] = grid[rr][cc]
        if not sc: return frame
        smr = min(rr for rr,cc in sc); smc = min(cc for rr,cc in sc)
        sxr = max(rr for rr,cc in sc); sxc = max(cc for rr,cc in sc)
        sh, sw = sxr-smr+1, sxc-smc+1
        shape = [[bg]*sw for _ in range(sh)]
        for rr, cc in sc: shape[rr-smr][cc-smc] = grid[rr][cc]
        lv = [frame[rr][0] for rr in range(fh) if frame[rr][0] != bg]
        rv = [frame[rr][fw-1] for rr in range(fh) if frame[rr][fw-1] != bg]
        lm = Counter(lv).most_common(1)[0][0]
        rm = Counter(rv).most_common(1)[0][0]
        sl = [shape[rr][cc] for rr in range(sh) for cc in range(sw//2) if shape[rr][cc] != bg]
        slm = Counter(sl).most_common(1)[0][0] if sl else bg
        if lm != rm and slm != lm:
            shape = [row[::-1] for row in shape]
        if len(set(lv)) == 1 and len(set(rv)) == 1:
            out = [[bg]*fw for _ in range(fh+2)]
            for rr in range(fh):
                out[rr+1][0] = frame[rr][0]; out[rr+1][fw-1] = frame[rr][fw-1]
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr][cc] != bg: out[rr+1][cc+1] = shape[rr][cc]
        else:
            out = [row[:] for row in frame]
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr][cc] != bg: out[rr+1][cc+1] = shape[rr][cc]
        return out

    def build_horiz(pair):
        (r1, s1, e1), (r2, s2, e2) = pair
        brt, brb = min(r1, r2), max(r1, r2)
        bcs = max(s1, s2); bce = min(e1, e2)
        fset = set()
        for cc in range(bcs, bce+1):
            fset.add((brt, cc)); fset.add((brb, cc))
        sc = [(rr, cc) for rr, cc in cells if (rr, cc) not in fset]
        fh, fw = brb-brt+1, bce-bcs+1
        frame = [[bg]*fw for _ in range(fh)]
        for rr, cc in cells:
            if (rr, cc) in fset:
                frame[rr-brt][cc-bcs] = grid[rr][cc]
        if not sc: return frame
        smr = min(rr for rr,cc in sc); smc = min(cc for rr,cc in sc)
        sxr = max(rr for rr,cc in sc); sxc = max(cc for rr,cc in sc)
        sh, sw = sxr-smr+1, sxc-smc+1
        shape = [[bg]*sw for _ in range(sh)]
        for rr, cc in sc: shape[rr-smr][cc-smc] = grid[rr][cc]
        tv = [frame[0][cc] for cc in range(fw) if frame[0][cc] != bg]
        bv = [frame[fh-1][cc] for cc in range(fw) if frame[fh-1][cc] != bg]
        tm = Counter(tv).most_common(1)[0][0]
        bm = Counter(bv).most_common(1)[0][0]
        st = [shape[rr][cc] for rr in range(sh//2) for cc in range(sw) if shape[rr][cc] != bg]
        stm = Counter(st).most_common(1)[0][0] if st else bg
        if tm != bm and stm != tm:
            shape = shape[::-1]
        pad_c = (fw - sw) // 2
        if len(set(tv)) == 1 and len(set(bv)) == 1:
            out = [[bg]*fw for _ in range(sh+2)]
            out[0] = frame[0][:]; out[sh+1] = frame[fh-1][:]
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr][cc] != bg: out[rr+1][pad_c+cc] = shape[rr][cc]
        else:
            out = [row[:] for row in frame]
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr][cc] != bg: out[rr+1][pad_c+cc] = shape[rr][cc]
        return out

    if best_vert_score >= best_horiz_score and best_vert_pair:
        return build_vert(best_vert_pair)
    elif best_horiz_pair:
        return build_horiz(best_horiz_pair)
    return grid

def transform(grid):
    return solve(grid)
