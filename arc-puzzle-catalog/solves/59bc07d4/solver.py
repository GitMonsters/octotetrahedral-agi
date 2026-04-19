def transform(grid):
    from collections import Counter
    
    H = len(grid)
    W = len(grid[0])
    
    # Find background (most frequent color)
    counts = Counter()
    for row in grid:
        for v in row:
            counts[v] += 1
    bg = counts.most_common(1)[0][0]
    
    # Collect non-background cells
    cells = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                cells[(r, c)] = grid[r][c]
    
    if not cells:
        return [row[:] for row in grid]
    
    def get_rotations(r, c, cr, cc):
        return [
            (r, c),
            (cr + cc - c, cc + r - cr),      # 90° CW
            (2 * cr - r, 2 * cc - c),         # 180°
            (cr - cc + c, cc - r + cr),        # 270° CW
        ]
    
    def try_center(cr, cc):
        pos_map = {}
        for (r, c), v in cells.items():
            for rr, rc in get_rotations(r, c, cr, cc):
                ri, ci = round(rr), round(rc)
                if abs(rr - ri) > 0.01 or abs(rc - ci) > 0.01:
                    return None
                if ri < 0 or ri >= H or ci < 0 or ci >= W:
                    return None
                if (ri, ci) in pos_map:
                    if pos_map[(ri, ci)] != v:
                        return None
                else:
                    pos_map[(ri, ci)] = v
        return pos_map
    
    # Search for the best center
    best = None
    best_key = None
    
    for cr2 in range(2 * H):
        for cc2 in range(2 * W):
            if cr2 % 2 != cc2 % 2:
                continue
            cr, cc = cr2 / 2.0, cc2 / 2.0
            pm = try_center(cr, cc)
            if pm is None:
                continue
            
            rs = [r for r, c in pm]
            cs = [c for r, c in pm]
            rmin, rmax = min(rs), max(rs)
            cmin, cmax = min(cs), max(cs)
            bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
            min_margin = min(rmin, H - 1 - rmax, cmin, W - 1 - cmax)
            
            key = (bbox_area, -min_margin)
            if best_key is None or key < best_key:
                best_key = key
                best = (cr, cc, pm)
    
    if best is None:
        return [row[:] for row in grid]
    
    cr, cc, pos_map = best
    out = [row[:] for row in grid]
    for (r, c), v in pos_map.items():
        out[r][c] = v
    
    return out
