def transform(grid):
    import numpy as np
    grid = [list(row) for row in grid]
    inp = np.array(grid)
    R, C = inp.shape
    
    # Find background (most common)
    from collections import Counter
    flat = [int(inp[r][c]) for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find horizontal divider lines (entire row of one non-bg color)
    h_lines = {}
    for r in range(R):
        vals = set(int(v) for v in inp[r])
        if len(vals) == 1 and int(list(vals)[0]) != bg:
            h_lines[r] = int(list(vals)[0])
    
    # Find vertical divider lines
    v_lines = {}
    for c in range(C):
        vals = set(int(v) for v in inp[:, c])
        if len(vals) == 1 and int(list(vals)[0]) != bg:
            v_lines[c] = int(list(vals)[0])
    
    # Collect colors that have lines
    line_colors = set(h_lines.values()) | set(v_lines.values())
    
    # Find noise cells (non-bg, not on any divider)
    noise = []
    for r in range(R):
        for c in range(C):
            v = int(inp[r][c])
            if v != bg and r not in h_lines and c not in v_lines:
                noise.append((r, c, v))
    
    # Start with bg everywhere, then add divider lines
    out = np.full((R, C), bg, dtype=int)
    for r, v in h_lines.items():
        out[r, :] = v
    for c, v in v_lines.items():
        out[:, c] = v
    
    # For each noise cell of a line color, snap to nearest line of that color
    for r, c, v in noise:
        if v not in line_colors:
            continue  # noise of non-line colors is removed
        
        # Find nearest line of this color
        best_dist = float('inf')
        best_pos = None
        
        for lr, lv in h_lines.items():
            if lv == v:
                d = abs(r - lr)
                if d < best_dist:
                    best_dist = d
                    if r < lr:
                        best_pos = (lr - 1, c)  # above
                    else:
                        best_pos = (lr + 1, c)  # below
        
        for lc, lv in v_lines.items():
            if lv == v:
                d = abs(c - lc)
                if d < best_dist:
                    best_dist = d
                    if c < lc:
                        best_pos = (r, lc - 1)  # left
                    else:
                        best_pos = (r, lc + 1)  # right
        
        if best_pos:
            pr, pc = best_pos
            if 0 <= pr < R and 0 <= pc < C:
                out[pr][pc] = v
    
    return out.tolist()
