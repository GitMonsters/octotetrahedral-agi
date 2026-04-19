def transform(input_grid):
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = Counter(v for r in input_grid for v in r).most_common(1)[0][0]
    
    zeros = sorted([(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == 0])
    if not zeros:
        return input_grid
    
    def scan_dir(r, c, dr, dc):
        dist = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            v = input_grid[nr][nc]
            if v != bg and v != 0:
                return dist, (nr, nc)
            dist += 1
            nr += dr
            nc += dc
        return dist, None
    
    rs = set(r for r, c in zeros)
    cs = set(c for r, c in zeros)
    
    if len(rs) == 1:
        horiz = True
        seg_r = list(rs)[0]
        seg_c_min, seg_c_max = min(cs), max(cs)
    else:
        horiz = False
        seg_c = list(cs)[0]
        seg_r_min, seg_r_max = min(rs), max(rs)
    
    def try_build(start_r, start_c, dr1, dc1, target):
        d1, w1 = scan_dir(start_r, start_c, dr1, dc1)
        if d1 == 0 or w1 is None:
            return None
        
        leg1 = []
        r, c = start_r, start_c
        for _ in range(d1):
            r += dr1; c += dc1
            leg1.append((r, c))
        corner1 = (r, c)
        
        # Pick perpendicular with longer scan for leg2
        perps = [(-dc1, dr1), (dc1, -dr1)]
        perp_scans = []
        for pdr, pdc in perps:
            d, w = scan_dir(corner1[0], corner1[1], pdr, pdc)
            perp_scans.append((d, w, pdr, pdc))
        
        # Sort by distance descending, prefer wall over no wall
        perp_scans.sort(key=lambda x: (x[1] is not None, x[0]), reverse=True)
        
        for d2, w2, dr2, dc2 in perp_scans:
            if d2 == 0:
                continue
            
            leg2 = []
            r2, c2 = corner1
            for _ in range(d2):
                r2 += dr2; c2 += dc2
                leg2.append((r2, c2))
            corner2 = (r2, c2)
            
            unique = len(zeros) + len(leg1) + len(leg2)
            needed = target - unique
            
            if needed < 0:
                continue
            
            # Leg3: pick perpendicular with best fit
            perps3 = [(-dc2, dr2), (dc2, -dr2)]
            for dr3, dc3 in perps3:
                d3, w3 = scan_dir(corner2[0], corner2[1], dr3, dc3)
                if d3 == 0 and needed > 0:
                    continue
                
                leg3_len = min(d3, needed) if needed > 0 else 0
                
                leg3 = []
                r3, c3 = corner2
                for _ in range(leg3_len):
                    r3 += dr3; c3 += dc3
                    if not (0 <= r3 < rows and 0 <= c3 < cols):
                        break
                    v = input_grid[r3][c3]
                    if v != bg and v != 0:
                        break
                    leg3.append((r3, c3))
                
                total = unique + len(leg3)
                if total == target:
                    return leg1 + leg2 + leg3
        
        return None
    
    # Try both targets: cols-1 and rows-1
    for target in [cols - 1, rows - 1]:
        all_paths = []
        if horiz:
            p = try_build(seg_r, seg_c_min, 0, -1, target)
            if p: all_paths.append(p)
            p = try_build(seg_r, seg_c_max, 0, 1, target)
            if p: all_paths.append(p)
        else:
            p = try_build(seg_r_min, seg_c, -1, 0, target)
            if p: all_paths.append(p)
            p = try_build(seg_r_max, seg_c, 1, 0, target)
            if p: all_paths.append(p)
        
        if all_paths:
            best = max(all_paths, key=len)
            output = [row[:] for row in input_grid]
            for r, c in best:
                output[r][c] = 0
            return output
    
    return input_grid
