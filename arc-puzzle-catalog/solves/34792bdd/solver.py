def transform(grid):
    from collections import Counter
    H, W = len(grid), len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    out = [row[:] for row in grid]
    
    input_colors = set(flat)
    fill = 8
    if fill in input_colors:
        for f in range(10):
            if f not in input_colors:
                fill = f
                break
    
    for col in range(W):
        vals = [grid[r][col] for r in range(H)]
        non_bg = [r for r in range(H) if vals[r] != bg]
        if not non_bg:
            continue
        
        segments = []
        start = non_bg[0]
        for i in range(1, len(non_bg)):
            if non_bg[i] != non_bg[i-1] + 1:
                segments.append((start, non_bg[i-1]))
                start = non_bg[i]
        segments.append((start, non_bg[-1]))
        
        for s, e in segments:
            touches_top = s == 0
            touches_bottom = e == H - 1
            if touches_top == touches_bottom:
                continue
            
            if touches_top:
                gap_clean = all(vals[r] == bg for r in range(e+1, H))
                gap_range = range(e+1, H)
            else:
                gap_clean = all(vals[r] == bg for r in range(0, s))
                gap_range = range(0, s)
            
            if not gap_clean:
                continue
            
            isolated = True
            for r in range(s, e+1):
                if col > 0 and grid[r][col-1] != bg:
                    isolated = False
                    break
                if col < W-1 and grid[r][col+1] != bg:
                    isolated = False
                    break
            
            if not isolated:
                continue
            
            for r in gap_range:
                out[r][col] = fill
    
    return out
