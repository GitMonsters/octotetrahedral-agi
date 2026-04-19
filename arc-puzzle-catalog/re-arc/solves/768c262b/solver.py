from collections import Counter
import math

def transform(grid):
    R, C = len(grid), len(grid[0])
    cnts = Counter(v for row in grid for v in row)
    bg = cnts.most_common(1)[0][0]

    h_lines = {}
    for r in range(R):
        vals = set(grid[r])
        if len(vals) == 1 and bg not in vals:
            h_lines[r] = list(vals)[0]
    v_lines = {}
    for c in range(C):
        vals = set(grid[r][c] for r in range(R))
        if len(vals) == 1 and bg not in vals:
            v_lines[c] = list(vals)[0]

    result = [[bg] * C for _ in range(R)]
    for lr, lc in h_lines.items():
        for c in range(C):
            result[lr][c] = lc
    for lc, lcolor in v_lines.items():
        for r in range(R):
            result[r][lc] = lcolor

    consumed = set()
    sorted_v = sorted(v_lines.keys())
    sorted_h = sorted(h_lines.keys())
    
    v_gap = (sorted_v[-1] - sorted_v[0]) if len(sorted_v) >= 2 else 999
    h_gap = (sorted_h[-1] - sorted_h[0]) if len(sorted_h) >= 2 else 999
    v_tight = v_gap <= 4
    h_tight = h_gap <= 4

    # Process V-lines
    for vidx, lc in enumerate(sorted_v):
        lcolor = v_lines[lc]
        is_first = (vidx == 0)
        next_lc = sorted_v[vidx+1] if vidx < len(sorted_v)-1 else C
        prev_lc = sorted_v[vidx-1] if vidx > 0 else -1
        
        for r in range(R):
            if r in h_lines:
                continue
            for side in ['left', 'right']:
                if side == 'left':
                    cols = range(lc-1, -1, -1)
                    adj_c = lc - 1
                    excl = set(range(prev_lc+1, lc)) if v_tight and vidx > 0 else set()
                else:
                    cols = range(lc+1, C)
                    adj_c = lc + 1
                    excl = set(range(lc+1, next_lc)) if v_tight and vidx < len(sorted_v)-1 else set()
                
                if adj_c < 0 or adj_c >= C:
                    continue
                
                scatter = [(c2, grid[r][c2]) for c2 in cols 
                          if c2 not in v_lines and grid[r][c2] != bg 
                          and (r,c2) not in consumed and c2 not in excl]
                
                match_n = sum(1 for _, v in scatter if v == lcolor)
                total_n = len(scatter)
                
                place = False
                if match_n > 0:
                    if is_first and v_tight:
                        place = match_n >= math.ceil(total_n / 2)
                    else:
                        place = True
                
                if place:
                    result[r][adj_c] = lcolor
                    if v_tight:
                        for c2, _ in scatter:
                            consumed.add((r, c2))
                    else:
                        for c2, v in scatter:
                            if v == lcolor:
                                consumed.add((r, c2))

    # Process H-lines
    for hidx, lr in enumerate(sorted_h):
        lcolor = h_lines[lr]
        is_first = (hidx == 0)
        
        for c in range(C):
            if c in v_lines:
                continue
            for side in ['above', 'below']:
                if side == 'above':
                    rows = range(lr-1, -1, -1)
                    adj_r = lr - 1
                else:
                    rows = range(lr+1, R)
                    adj_r = lr + 1
                
                if adj_r < 0 or adj_r >= R:
                    continue
                
                scatter = [(r2, grid[r2][c]) for r2 in rows 
                          if r2 not in h_lines and grid[r2][c] != bg 
                          and (r2,c) not in consumed]
                
                # H-dedup: skip if same-color neighbor to the left
                dedup = []
                for r2, v in scatter:
                    if c > 0 and c - 1 not in v_lines and grid[r2][c-1] == v and grid[r2][c-1] != bg:
                        continue
                    dedup.append((r2, v))
                
                match_n = sum(1 for _, v in dedup if v == lcolor)
                total_n = len(dedup)
                
                place = False
                if match_n > 0:
                    if is_first:
                        place = total_n != 2
                    else:
                        place = True
                
                if place:
                    result[adj_r][c] = lcolor
                    # Consume only matching for h_lines
                    for r2, v in scatter:
                        if v == lcolor:
                            consumed.add((r2, c))

    return result
