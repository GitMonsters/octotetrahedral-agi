from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]
    output = [row[:] for row in grid]
    
    # Find FULL stripe row (all non-bg, exactly 2 colors)
    stripe_row = None
    stripe_row_colors = None
    for r in range(H):
        vals = set(grid[r])
        if bg not in vals and len(vals) == 2:
            stripe_row = r
            stripe_row_colors = vals
            break
    
    # Find FULL stripe col (all non-bg, exactly 2 colors) 
    stripe_col = None
    stripe_col_colors = None
    for c in range(W):
        vals = set(grid[r][c] for r in range(H))
        if bg not in vals and len(vals) == 2:
            stripe_col = c
            stripe_col_colors = vals
            break
    
    # Also find single-color full rows/cols
    mono_row = None
    for r in range(H):
        vals = set(grid[r])
        if bg not in vals and len(vals) == 1:
            mono_row = r
            break
    
    mono_col = None
    for c in range(W):
        vals = set(grid[r][c] for r in range(H))
        if bg not in vals and len(vals) == 1:
            mono_col = c
            break
    
    if stripe_row is not None:
        # Stripe is a row → extend partial rows using stripe's column pattern
        sa, sb = sorted(stripe_row_colors)
        pattern = grid[stripe_row]  # value at each col = sa or sb
        
        for r in range(H):
            if r == stripe_row:
                continue
            non_bg = [(c, grid[r][c]) for c in range(W) if grid[r][c] != bg]
            if len(non_bg) == 0:
                continue
            
            # Determine mapping from stripe values to row values
            color_map = {}
            for c, v in non_bg:
                sv = pattern[c]
                if sv not in color_map:
                    color_map[sv] = v
            
            if len(color_map) < 2:
                # Only one stripe color mapped; the other maps to bg
                for sv in stripe_row_colors:
                    if sv not in color_map:
                        color_map[sv] = bg
            
            # Extend row
            for c in range(W):
                output[r][c] = color_map.get(pattern[c], bg)
    
    elif stripe_col is not None:
        # Stripe is a column → extend partial columns using stripe's row pattern
        sa, sb = sorted(stripe_col_colors)
        pattern = [grid[r][stripe_col] for r in range(H)]  # value at each row
        
        for c in range(W):
            if c == stripe_col:
                continue
            non_bg = [(r, grid[r][c]) for r in range(H) if grid[r][c] != bg]
            if len(non_bg) == 0:
                continue
            
            color_map = {}
            for r, v in non_bg:
                sv = pattern[r]
                if sv not in color_map:
                    color_map[sv] = v
            
            if len(color_map) < 2:
                for sv in stripe_col_colors:
                    if sv not in color_map:
                        color_map[sv] = bg
            
            for r in range(H):
                output[r][c] = color_map.get(pattern[r], bg)
    
    elif mono_row is not None:
        # Single-color full row: derive binary column template using row-to-col mapping
        mono_val = grid[mono_row][0]
        D = W - H  # width minus height; determines column offset per active row

        # Find full columns (cols with >1 non-bg value outside mono_row)
        full_col_patterns = {}
        for c in range(W):
            vals = {r: grid[r][c] for r in range(H)
                    if r != mono_row and grid[r][c] != bg}
            if len(vals) > 1:
                full_col_patterns[c] = vals

        if full_col_patterns:
            # Classify full cols: A-type contains mono_val, B-type does not
            a_cols = [c for c, vals in full_col_patterns.items()
                      if mono_val in set(vals.values())]
            b_cols = [c for c, vals in full_col_patterns.items()
                      if mono_val not in set(vals.values())]

            if a_cols and b_cols:
                # Compute B_positions using active-row → column formula
                partial_rows = [r for r in range(H)
                                if r != mono_row and
                                any(grid[r][c] != bg for c in range(W))]
                active_rows = partial_rows + [mono_row]

                B_positions = set(b_cols)  # visible B-type full cols are B
                for r in active_rows:
                    if r >= mono_row:
                        # mono row and below: B at {r, r+D}
                        B_positions.add(r)
                        if 0 <= r + D < W:
                            B_positions.add(r + D)
                    elif abs(r - mono_row) <= max(1, abs(D)):
                        # adjacent above mono: B at {W-r}
                        if 0 <= W - r < W:
                            B_positions.add(W - r)
                    else:
                        # non-adjacent above mono: B at {r, r+D, W-r}
                        B_positions.add(r)
                        if 0 <= r + D < W:
                            B_positions.add(r + D)
                        if 0 <= W - r < W:
                            B_positions.add(W - r)

                a_ref = a_cols[0]
                b_ref = b_cols[0]

                for r in range(H):
                    if r == mono_row:
                        continue
                    if not any(grid[r][c] != bg for c in range(W)):
                        continue
                    a_color = full_col_patterns[a_ref].get(r)
                    b_color = full_col_patterns[b_ref].get(r)
                    if a_color is None or b_color is None:
                        continue
                    for c in range(W):
                        output[r][c] = b_color if c in B_positions else a_color
    
    return output
