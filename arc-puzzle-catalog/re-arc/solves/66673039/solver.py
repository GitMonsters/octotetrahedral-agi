from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    # Find divider
    divider_dir = divider_pos = divider_color = None
    for r in range(H):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            divider_dir, divider_pos, divider_color = 'h', r, grid[r][0]
            break
    if divider_dir is None:
        for c in range(W):
            vals = set(grid[r][c] for r in range(H))
            if len(vals) == 1 and list(vals)[0] != bg:
                divider_dir, divider_pos, divider_color = 'v', c, list(vals)[0]
                break
    if divider_dir is None:
        return [row[:] for row in grid]
    
    # Determine sides
    if divider_dir == 'h':
        above = sum(1 for r in range(divider_pos) for c in range(W)
                    if grid[r][c] != bg and grid[r][c] != divider_color)
        below = sum(1 for r in range(divider_pos+1, H) for c in range(W)
                    if grid[r][c] != bg and grid[r][c] != divider_color)
        pat_side = 'above' if above >= below else 'below'
        blank_r = range(divider_pos+1, H) if pat_side == 'above' else range(divider_pos)
        blank_c = range(W)
        pat_r = range(divider_pos) if pat_side == 'above' else range(divider_pos+1, H)
        pat_c = range(W)
    else:
        left = sum(1 for r in range(H) for c in range(divider_pos)
                   if grid[r][c] != bg and grid[r][c] != divider_color)
        right = sum(1 for r in range(H) for c in range(divider_pos+1, W)
                    if grid[r][c] != bg and grid[r][c] != divider_color)
        pat_side = 'left' if left >= right else 'right'
        blank_c = range(divider_pos+1, W) if pat_side == 'left' else range(divider_pos)
        blank_r = range(H)
        pat_c = range(divider_pos) if pat_side == 'left' else range(divider_pos+1, W)
        pat_r = range(H)
    
    # Extract stamp
    sr = [(r,c) for r in pat_r for c in pat_c 
          if grid[r][c] != bg and grid[r][c] != divider_color]
    if not sr:
        return [row[:] for row in grid]
    min_r = min(r for r,c in sr)
    max_r = max(r for r,c in sr)
    min_c = min(c for r,c in sr)
    max_c = max(c for r,c in sr)
    sh = max_r - min_r + 1
    sw = max_c - min_c + 1
    stamp = [[grid[min_r+r][min_c+c] for c in range(sw)] for r in range(sh)]
    center_r, center_c = sh // 2, sw // 2
    
    # Find markers
    markers = []
    for r in blank_r:
        for c in blank_c:
            if grid[r][c] != bg and grid[r][c] != divider_color:
                markers.append((r, c))
    
    result = [row[:] for row in grid]
    
    if markers:
        for mr, mc in markers:
            r0 = mr - center_r
            c0 = mc - center_c
            for r in range(sh):
                for c in range(sw):
                    rr, cc = r0 + r, c0 + c
                    if 0 <= rr < H and 0 <= cc < W:
                        if stamp[r][c] != bg:
                            result[rr][cc] = stamp[r][c]
    else:
        # No markers: tile using self-similar pattern
        blank_r_start = min(blank_r)
        blank_c_start = min(blank_c)
        blank_h = max(blank_r) - blank_r_start + 1
        blank_w = max(blank_c) - blank_c_start + 1
        n_rows = blank_h // sh
        n_cols = blank_w // sw
        
        # Build non-bg mask
        nonbg = set()
        for r in range(sh):
            for c in range(sw):
                if stamp[r][c] != bg:
                    nonbg.add((r, c))
        
        # 180° rotate + transpose to get tiling mask
        tiling = set()
        for r, c in nonbg:
            nr, nc = sh - 1 - r, sw - 1 - c
            tr, tc = nc, nr  # transpose
            if 0 <= tr < n_rows and 0 <= tc < n_cols:
                tiling.add((tr, tc))
        
        # Determine seed row (row in tiling grid corresponding to original stamp)
        if divider_dir == 'v':
            seed_tiling_row = (min_r - blank_r_start) // sh
        else:
            seed_tiling_row = (min_r - blank_r_start) // sh
        
        # Fix seed row: replace with endpoints
        tiling = set((tr, tc) for tr, tc in tiling if tr != seed_tiling_row)
        tiling.add((seed_tiling_row, 0))
        tiling.add((seed_tiling_row, n_cols - 1))
        
        # Remove position corresponding to stamp center-right 2-cell
        # This is the cell that maps to a position that breaks the pattern
        for r, c in nonbg:
            if c == sw - 1 and r == center_r:  # center row, rightmost col
                nr, nc = sh - 1 - r, sw - 1 - c
                tr, tc = nc, nr
                tiling.discard((tr, tc))
        
        # Place stamps at tiling positions
        for tr, tc in tiling:
            r0 = blank_r_start + tr * sh
            c0 = blank_c_start + tc * sw
            for r in range(sh):
                for c in range(sw):
                    rr, cc = r0 + r, c0 + c
                    if 0 <= rr < H and 0 <= cc < W:
                        if stamp[r][c] != bg:
                            result[rr][cc] = stamp[r][c]
    
    return result
