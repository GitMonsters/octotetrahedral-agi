from collections import Counter

def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0])
    
    # 1. Background color (most common)
    counts = Counter(v for row in input_grid for v in row)
    bg = counts.most_common(1)[0][0]
    
    # 2. Find frame color: non-bg with single-color holes in bbox, most solid borders
    non_bg = [c for c in counts if c != bg]
    
    best_frame = None
    best_score = -1
    for color in non_bg:
        cells = set()
        for r in range(R):
            for c in range(C):
                if input_grid[r][c] == color:
                    cells.add((r, c))
        rmin = min(r for r, c in cells)
        rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells)
        cmax = max(c for r, c in cells)
        
        hole_colors = set()
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if input_grid[r][c] != color:
                    hole_colors.add(input_grid[r][c])
        if len(hole_colors) != 1:
            continue
        
        score = 0
        if all((rmin, c) in cells for c in range(cmin, cmax + 1)):
            score += 1
        if all((rmax, c) in cells for c in range(cmin, cmax + 1)):
            score += 1
        if all((r, cmin) in cells for r in range(rmin, rmax + 1)):
            score += 1
        if all((r, cmax) in cells for r in range(rmin, rmax + 1)):
            score += 1
        
        if score > best_score:
            best_score = score
            best_frame = color
    
    frame_color = best_frame
    
    # 3. Frame cells and bbox
    frame_cells = [(r, c) for r in range(R) for c in range(C)
                   if input_grid[r][c] == frame_color]
    fr1 = min(r for r, c in frame_cells)
    fr2 = max(r for r, c in frame_cells)
    fc1 = min(c for r, c in frame_cells)
    fc2 = max(c for r, c in frame_cells)
    
    # 4. Hole cells inside frame bbox
    hole_cells = [(r, c) for r in range(fr1, fr2 + 1)
                  for c in range(fc1, fc2 + 1)
                  if input_grid[r][c] != frame_color]
    inner_color = input_grid[hole_cells[0][0]][hole_cells[0][1]]
    hr1 = min(r for r, c in hole_cells)
    hr2 = max(r for r, c in hole_cells)
    hc1 = min(c for r, c in hole_cells)
    hc2 = max(c for r, c in hole_cells)
    
    # 5. Find individual hole ranges (handles multi-hole frames)
    def get_ranges(values):
        vals = sorted(set(values))
        ranges = []
        i = 0
        while i < len(vals):
            start = vals[i]
            while i + 1 < len(vals) and vals[i + 1] == vals[i] + 1:
                i += 1
            ranges.append((start, vals[i]))
            i += 1
        return ranges
    
    hole_col_ranges = get_ranges([c for r, c in hole_cells])
    hole_row_ranges = get_ranges([r for r, c in hole_cells])
    
    ind_hole_w = hole_col_ranges[0][1] - hole_col_ranges[0][0] + 1
    ind_hole_h = hole_row_ranges[0][1] - hole_row_ranges[0][0] + 1
    
    # Border thicknesses
    top_b = hr1 - fr1
    bot_b = fr2 - hr2
    left_b = hc1 - fc1
    right_b = fc2 - hc2
    
    # Inter-hole border or max edge border
    if len(hole_col_ranges) > 1:
        h_border = hole_col_ranges[1][0] - hole_col_ranges[0][1] - 1
    else:
        h_border = max(left_b, right_b)
    
    if len(hole_row_ranges) > 1:
        v_border = hole_row_ranges[1][0] - hole_row_ranges[0][1] - 1
    else:
        v_border = max(top_b, bot_b)
    
    step_v = v_border + ind_hole_h
    step_h = h_border + ind_hole_w
    
    # 6. Direction logic
    if inner_color == bg:
        move_v = True
        move_h = True
        dir_v = 1
        dir_h = 1
    else:
        has_v_sat = False
        has_h_sat = False
        for r in range(R):
            for c in range(C):
                if input_grid[r][c] == inner_color:
                    if not (fr1 <= r <= fr2 and fc1 <= c <= fc2):
                        if any(hs <= c <= he for hs, he in hole_col_ranges):
                            has_v_sat = True
                        if any(hs <= r <= he for hs, he in hole_row_ranges):
                            has_h_sat = True
        
        move_v = has_v_sat
        move_h = has_h_sat
        if not move_v and not move_h:
            move_v = move_h = True
        
        dir_v = 1
        dir_h = 1
        
        if move_v and step_v > 0:
            above_start = hole_row_ranges[0][0] - step_v
            above_clipped = above_start < 0
            below_end = hole_row_ranges[0][0] + step_v + ind_hole_h - 1
            below_clipped = below_end >= R
            if above_clipped and not below_clipped:
                dir_v = 1
            elif below_clipped and not above_clipped:
                dir_v = -1
        
        if move_h and step_h > 0:
            left_start = hole_col_ranges[0][0] - step_h
            left_clipped = left_start < 0
            right_end = hole_col_ranges[0][0] + step_h + ind_hole_w - 1
            right_clipped = right_end >= C
            if left_clipped and not right_clipped:
                dir_h = 1
            elif right_clipped and not left_clipped:
                dir_h = -1
    
    dv = step_v * dir_v if move_v else 0
    dh = step_h * dir_h if move_h else 0
    
    # 7. Build output
    output = [row[:] for row in input_grid]
    
    # Erase old frame
    for r, c in frame_cells:
        output[r][c] = bg
    
    # Draw new frame (shifted, clipped to grid)
    for r, c in frame_cells:
        nr, nc = r + dv, c + dh
        if 0 <= nr < R and 0 <= nc < C:
            output[nr][nc] = frame_color
    
    return output
