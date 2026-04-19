def transform(grid):
    from collections import Counter
    H, W = len(grid), len(grid[0])
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    markers = set()
    marker_color = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                markers.add((r, c))
                marker_color = grid[r][c]
    
    if not markers:
        return [row[:] for row in grid]
    
    min_r = min(r for r, c in markers)
    max_r = max(r for r, c in markers)
    min_c = min(c for r, c in markers)
    max_c = max(c for r, c in markers)
    
    input_colors = set(flat)
    fill = 3
    if fill in input_colors:
        for f in range(10):
            if f not in input_colors:
                fill = f
                break
    
    out = [row[:] for row in grid]
    
    # Fill interior of bounding box
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if (r, c) not in markers:
                out[r][c] = fill
    
    # Extend through top boundary gaps
    for c in range(min_c, max_c + 1):
        if (min_r, c) not in markers:
            for r in range(min_r - 1, -1, -1):
                out[r][c] = fill
    
    # Extend through bottom boundary gaps
    for c in range(min_c, max_c + 1):
        if (max_r, c) not in markers:
            for r in range(max_r + 1, H):
                out[r][c] = fill
    
    # Extend through left boundary gaps
    for r in range(min_r, max_r + 1):
        if (r, min_c) not in markers:
            for c in range(min_c - 1, -1, -1):
                out[r][c] = fill
    
    # Extend through right boundary gaps
    for r in range(min_r, max_r + 1):
        if (r, max_c) not in markers:
            for c in range(max_c + 1, W):
                out[r][c] = fill
    
    return out
