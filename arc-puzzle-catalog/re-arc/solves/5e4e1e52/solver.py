def transform(grid):
    """
    ARC puzzle 5e4e1e52:
    1. Find bracket shape (rectangular frame with one side having a gap)
    2. Fill interior with color 2
    3. From gap: draw diagonal lines from corners + fill gap rows/cols straight to edge
    """
    import copy
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    color_counts = {}
    for row in grid:
        for c in row:
            color_counts[c] = color_counts.get(c, 0) + 1
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find the bracket color
    bracket_color = None
    for row in grid:
        for c in row:
            if c != bg_color:
                bracket_color = c
                break
        if bracket_color:
            break
    
    # Find bounding box of bracket
    min_r, max_r, min_c, max_c = h, 0, w, 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bracket_color:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    result = copy.deepcopy(grid)
    
    # Fill the interior of the bracket with color 2
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if result[r][c] == bg_color:
                result[r][c] = 2
    
    # Find the gaps on each side
    left_gap_rows = [r for r in range(min_r, max_r + 1) if grid[r][min_c] == bg_color]
    right_gap_rows = [r for r in range(min_r, max_r + 1) if grid[r][max_c] == bg_color]
    top_gap_cols = [c for c in range(min_c, max_c + 1) if grid[min_r][c] == bg_color]
    bottom_gap_cols = [c for c in range(min_c, max_c + 1) if grid[max_r][c] == bg_color]
    
    # Left gap: project to left
    if left_gap_rows:
        gap_top = min(left_gap_rows)
        gap_bottom = max(left_gap_rows)
        
        # Fill gap rows straight left
        for r in range(gap_top, gap_bottom + 1):
            for c in range(0, min_c):
                result[r][c] = 2
        
        # Diagonal from top corner going up-left
        r, c = gap_top - 1, min_c - 1
        while r >= 0 and c >= 0:
            result[r][c] = 2
            r -= 1
            c -= 1
        
        # Diagonal from bottom corner going down-left
        r, c = gap_bottom + 1, min_c - 1
        while r < h and c >= 0:
            result[r][c] = 2
            r += 1
            c -= 1
    
    # Right gap: project to right
    if right_gap_rows:
        gap_top = min(right_gap_rows)
        gap_bottom = max(right_gap_rows)
        
        # Fill gap rows straight right
        for r in range(gap_top, gap_bottom + 1):
            for c in range(max_c + 1, w):
                result[r][c] = 2
        
        # Diagonal from top corner going up-right
        r, c = gap_top - 1, max_c + 1
        while r >= 0 and c < w:
            result[r][c] = 2
            r -= 1
            c += 1
        
        # Diagonal from bottom corner going down-right
        r, c = gap_bottom + 1, max_c + 1
        while r < h and c < w:
            result[r][c] = 2
            r += 1
            c += 1
    
    # Top gap: project upward
    if top_gap_cols:
        gap_left = min(top_gap_cols)
        gap_right = max(top_gap_cols)
        
        # Fill gap cols straight up
        for c in range(gap_left, gap_right + 1):
            for r in range(0, min_r):
                result[r][c] = 2
        
        # Diagonal from left corner going up-left
        r, c = min_r - 1, gap_left - 1
        while r >= 0 and c >= 0:
            result[r][c] = 2
            r -= 1
            c -= 1
        
        # Diagonal from right corner going up-right
        r, c = min_r - 1, gap_right + 1
        while r >= 0 and c < w:
            result[r][c] = 2
            r -= 1
            c += 1
    
    # Bottom gap: project downward
    if bottom_gap_cols:
        gap_left = min(bottom_gap_cols)
        gap_right = max(bottom_gap_cols)
        
        # Fill gap cols straight down
        for c in range(gap_left, gap_right + 1):
            for r in range(max_r + 1, h):
                result[r][c] = 2
        
        # Diagonal from left corner going down-left
        r, c = max_r + 1, gap_left - 1
        while r < h and c >= 0:
            result[r][c] = 2
            r += 1
            c -= 1
        
        # Diagonal from right corner going down-right
        r, c = max_r + 1, gap_right + 1
        while r < h and c < w:
            result[r][c] = 2
            r += 1
            c += 1
    
    return result
