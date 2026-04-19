def transform(input_grid):
    from collections import Counter
    
    inp = input_grid
    H, W = len(inp), len(inp[0])
    flat = [c for row in inp for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find non-bg rectangular region bounding box
    non_bg = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != bg]
    if not non_bg:
        return [row[:] for row in inp]
    
    r_min = min(r for r, c in non_bg)
    r_max = max(r for r, c in non_bg)
    c_min = min(c for r, c in non_bg)
    c_max = max(c for r, c in non_bg)
    
    # Find frame and interior colors
    non_bg_colors = set(inp[r][c] for r, c in non_bg)
    
    # Determine frame (outermost color) vs interior
    if len(non_bg_colors) >= 2:
        edge_counter = Counter()
        for c in range(c_min, c_max + 1):
            edge_counter[inp[r_min][c]] += 1
            edge_counter[inp[r_max][c]] += 1
        for r in range(r_min + 1, r_max):
            edge_counter[inp[r][c_min]] += 1
            edge_counter[inp[r][c_max]] += 1
        edge_counter.pop(bg, None)
        
        frame_color = edge_counter.most_common(1)[0][0]
        interior_color = (non_bg_colors - {frame_color}).pop()
    else:
        # 2-color case: frame == interior
        frame_color = non_bg_colors.pop()
        interior_color = frame_color
    
    # Detect protrusions at distance 1 from frame outer edge
    if interior_color != frame_color:
        top_cols = set()
        bottom_cols = set()
        left_rows = set()
        right_rows = set()
        
        if r_min + 1 <= r_max:
            top_cols = {c for c in range(c_min, c_max + 1)
                        if inp[r_min + 1][c] == interior_color}
        if r_max - 1 >= r_min:
            bottom_cols = {c for c in range(c_min, c_max + 1)
                           if inp[r_max - 1][c] == interior_color}
        if c_min + 1 <= c_max:
            left_rows = {r for r in range(r_min, r_max + 1)
                         if inp[r][c_min + 1] == interior_color}
        if c_max - 1 >= c_min:
            right_rows = {r for r in range(r_min, r_max + 1)
                          if inp[r][c_max - 1] == interior_color}
        
        extend_cols = top_cols | bottom_cols
        extend_rows = left_rows | right_rows
    else:
        # 2-color: generate protrusion pattern from background spaces
        top_space = r_min
        bottom_space = H - 1 - r_max
        left_space = c_min
        right_space = W - 1 - c_max
        
        body_h = (r_max - 2) - (r_min + 2) + 1
        body_w = (c_max - 2) - (c_min + 2) + 1
        body_r_start = r_min + 2
        body_c_start = c_min + 2
        
        # Protrusion count per side = bg_space // 2
        top_count = top_space // 2
        bottom_count = bottom_space // 2
        left_count = left_space // 2
        right_count = right_space // 2
        
        def distribute_positions(count_a, count_b, body_dim):
            """Distribute protrusion positions for a pair of opposite sides."""
            total = count_a + count_b
            if total == 0:
                return []
            if count_a == 0:
                # Center count_b positions
                start = (body_dim - count_b + 1) // 2
                return list(range(start, start + count_b))
            if count_b == 0:
                start = (body_dim - count_a + 1) // 2
                return list(range(start, start + count_a))
            # Side A: first count_a consecutive from start
            pos_a = list(range(count_a))
            # Side B: starting from count_a, step = count_a
            pos_b = [count_a + i * count_a for i in range(count_b)
                      if count_a + i * count_a < body_dim]
            return sorted(set(pos_a + pos_b))
        
        row_positions = distribute_positions(left_count, right_count, body_h)
        col_positions = distribute_positions(top_count, bottom_count, body_w)
        
        extend_rows = {body_r_start + p for p in row_positions}
        extend_cols = {body_c_start + p for p in col_positions}
    
    # Body region (distance 2 from all frame edges)
    body_r_min = r_min + 2
    body_r_max = r_max - 2
    body_c_min = c_min + 2
    body_c_max = c_max - 2
    
    # Build output
    out = [[bg] * W for _ in range(H)]
    
    # Fill frame rectangle with frame color
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            out[r][c] = frame_color
    
    # Non-protrusion intersections within body keep interior color
    if interior_color != frame_color:
        for r in range(body_r_min, body_r_max + 1):
            for c in range(body_c_min, body_c_max + 1):
                if r not in extend_rows and c not in extend_cols:
                    out[r][c] = interior_color
    
    # Extend protrusion columns into background (up and down)
    for c in extend_cols:
        for r in range(0, r_min):
            out[r][c] = interior_color
        for r in range(r_max + 1, H):
            out[r][c] = interior_color
    
    # Extend protrusion rows into background (left and right)
    for r in extend_rows:
        for c in range(0, c_min):
            out[r][c] = interior_color
        for c in range(c_max + 1, W):
            out[r][c] = interior_color
    
    return out
