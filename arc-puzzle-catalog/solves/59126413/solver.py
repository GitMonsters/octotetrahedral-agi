def transform(grid):
    H = len(grid)
    W = len(grid[0])
    from collections import Counter
    
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common()[0][0]
    out = [[bg]*W for _ in range(H)]
    
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                non_bg[(r,c)] = grid[r][c]
    
    if not non_bg:
        return out
    
    # Detect wall pairs on opposing edges
    edge_pixels = {
        'left': [(r, 0) for r in range(H) if (r, 0) in non_bg],
        'right': [(r, W-1) for r in range(H) if (r, W-1) in non_bg],
        'top': [(0, c) for c in range(W) if (0, c) in non_bg],
        'bottom': [(H-1, c) for c in range(W) if (H-1, c) in non_bg],
    }
    
    wall_pair = None
    if len(edge_pixels['left']) >= 2 and len(edge_pixels['right']) >= 2:
        wall_pair = ('left', 'right')
    elif len(edge_pixels['top']) >= 2 and len(edge_pixels['bottom']) >= 2:
        wall_pair = ('top', 'bottom')
    
    wall_positions = set()
    if wall_pair:
        for name in wall_pair:
            wall_positions.update(edge_pixels[name])
    
    dots = [(r,c,non_bg[(r,c)]) for (r,c) in non_bg if (r,c) not in wall_positions]
    
    # Copy wall pixels to output
    for (r,c) in wall_positions:
        out[r][c] = non_bg[(r,c)]
    
    MARKER = 5
    if not dots:
        return out
    dot_color = dots[0][2]
    
    if wall_pair == ('left', 'right'):
        left_rows = sorted([r for r,c in edge_pixels['left']])
        right_rows = sorted([r for r,c in edge_pixels['right']])
        
        # Determine target wall (the one dots align with by row range)
        target = None
        for dr, dc, dv in dots:
            if right_rows[0] <= dr <= right_rows[-1]:
                target = 'right'; break
            if left_rows[0] <= dr <= left_rows[-1]:
                target = 'left'; break
        
        if target == 'right':
            target_rows, target_col = right_rows, W - 1
            opposite_rows, opposite_col = left_rows, 0
        else:
            target_rows, target_col = left_rows, 0
            opposite_rows, opposite_col = right_rows, W - 1
        
        target_start = target_rows[0]
        opposite_start = opposite_rows[0]
        
        for dr, dc, dv in dots:
            out[dr][dc] = MARKER
            # Primary horizontal stripe from dot to wall
            if target_col > dc:
                for c in range(dc + 1, target_col):
                    out[dr][c] = dot_color
            else:
                for c in range(target_col + 1, dc):
                    out[dr][c] = dot_color
            # Reflected stripe
            index = dr - target_start
            reflected_row = opposite_start + index
            if 0 <= reflected_row < H:
                for c in range(W):
                    if c != opposite_col and out[reflected_row][c] == bg:
                        out[reflected_row][c] = dot_color
    
    elif wall_pair == ('top', 'bottom'):
        top_cols = sorted([c for r,c in edge_pixels['top']])
        bottom_cols = sorted([c for r,c in edge_pixels['bottom']])
        
        target = None
        for dr, dc, dv in dots:
            if bottom_cols[0] <= dc <= bottom_cols[-1]:
                target = 'bottom'; break
            if top_cols[0] <= dc <= top_cols[-1]:
                target = 'top'; break
        
        if target == 'bottom':
            target_cols, target_row = bottom_cols, H - 1
            opposite_cols, opposite_row = top_cols, 0
        else:
            target_cols, target_row = top_cols, 0
            opposite_cols, opposite_row = bottom_cols, H - 1
        
        target_start = target_cols[0]
        opposite_start = opposite_cols[0]
        
        for dr, dc, dv in dots:
            out[dr][dc] = MARKER
            if target_row > dr:
                for r in range(dr + 1, target_row):
                    out[r][dc] = dot_color
            else:
                for r in range(target_row + 1, dr):
                    out[r][dc] = dot_color
            index = dc - target_start
            reflected_col = opposite_start + index
            if 0 <= reflected_col < W:
                for r in range(H):
                    if r != opposite_row and out[r][reflected_col] == bg:
                        out[r][reflected_col] = dot_color
    
    else:
        # No walls - vertical projection downward, reflect leftward
        min_col = min(dc for dr, dc, dv in dots)
        for dr, dc, dv in dots:
            out[dr][dc] = MARKER
            for r in range(dr + 1, H - 1):
                out[r][dc] = dot_color
            reflected_col = dc - min_col
            for r in range(1, H):
                if out[r][reflected_col] == bg:
                    out[r][reflected_col] = dot_color
    
    return out
