def transform(grid):
    from collections import Counter
    H = len(grid)
    W = len(grid[0])
    
    # Find background color (most common)
    counts = Counter(v for row in grid for v in row)
    bg = counts.most_common(1)[0][0]
    
    # Collect non-background pixel positions
    non_bg = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    
    # Determine which diagonal: compute avg distance of non-bg pixels to each
    if non_bg:
        avg_main = sum(abs(r - c) for r, c in non_bg) / len(non_bg)
        avg_anti = sum(abs(r + c - (H - 1)) for r, c in non_bg) / len(non_bg)
        use_main = avg_main <= avg_anti
    else:
        # Default for uniform grids
        use_main = True
    
    # Build output: all 0, with chosen diagonal in color 8
    out = [[0] * W for _ in range(H)]
    for i in range(min(H, W)):
        if use_main:
            out[i][i] = 8
        else:
            out[i][W - 1 - i] = 8
    
    return out
