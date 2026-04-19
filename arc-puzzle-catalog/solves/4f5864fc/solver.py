from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    nonbg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                nonbg[(r,c)] = grid[r][c]
    
    # 8-connected components
    visited = set()
    components = []
    for pos in nonbg:
        if pos in visited:
            continue
        comp = []
        stack = [pos]
        visited.add(pos)
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in nonbg and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        components.append(comp)
    
    def get_line_segments(cells):
        cell_set = set(cells)
        h_segs = []
        v_segs = []
        row_cells = {}
        for r, c in cells:
            row_cells.setdefault(r, []).append(c)
        for r, cs in row_cells.items():
            cs.sort()
            start = cs[0]
            for i in range(1, len(cs)):
                if cs[i] != cs[i-1] + 1:
                    if cs[i-1] - start >= 1:
                        h_segs.append((r, start, cs[i-1], nonbg[(r, start)]))
                    start = cs[i]
            if cs[-1] - start >= 1:
                h_segs.append((r, start, cs[-1], nonbg[(r, start)]))
        col_cells = {}
        for r, c in cells:
            col_cells.setdefault(c, []).append(r)
        for c, rs in col_cells.items():
            rs.sort()
            start = rs[0]
            for i in range(1, len(rs)):
                if rs[i] != rs[i-1] + 1:
                    if rs[i-1] - start >= 1:
                        v_segs.append((c, start, rs[i-1], nonbg[(start, c)]))
                    start = rs[i]
            if rs[-1] - start >= 1:
                v_segs.append((c, start, rs[-1], nonbg[(start, c)]))
        return h_segs, v_segs
    
    def compute_rectangle(h_segs, v_segs):
        """Compute frame rectangle from line segments. Returns (top, bot, left, right) or None."""
        h_rows = sorted(set(seg[0] for seg in h_segs)) if h_segs else []
        v_cols = sorted(set(seg[0] for seg in v_segs)) if v_segs else []
        
        if not h_rows and not v_cols:
            return None
        
        # Top/bottom
        if len(h_rows) >= 2:
            top_row = min(h_rows)
            bot_row = max(h_rows)
        elif len(h_rows) == 1:
            h_row = h_rows[0]
            if v_segs:
                v_min_r = min(seg[1] for seg in v_segs)
                v_max_r = max(seg[2] for seg in v_segs)
                if h_row <= v_min_r:
                    top_row = h_row
                    bot_row = v_max_r + 1
                else:
                    top_row = v_min_r - 1
                    bot_row = h_row
            else:
                return None
        else:
            if v_segs:
                top_row = min(seg[1] for seg in v_segs) - 1
                bot_row = max(seg[2] for seg in v_segs) + 1
            else:
                return None
        
        # Left/right
        if len(v_cols) >= 2:
            left_col = min(v_cols)
            right_col = max(v_cols)
        elif len(v_cols) == 1:
            v_col = v_cols[0]
            if h_segs:
                h_min_c = min(seg[1] for seg in h_segs)
                h_max_c = max(seg[2] for seg in h_segs)
                if v_col <= h_min_c:
                    left_col = v_col
                    right_col = h_max_c + 1
                else:
                    left_col = h_min_c - 1
                    right_col = v_col
            else:
                return None
        else:
            if h_segs:
                left_col = min(seg[1] for seg in h_segs) - 1
                right_col = max(seg[2] for seg in h_segs) + 1
            else:
                return None
        
        return (top_row, bot_row, left_col, right_col)
    
    def get_border_colors(h_segs, v_segs, rect):
        top_row, bot_row, left_col, right_col = rect
        tc, bc, lc, rc = bg, bg, bg, bg
        for r, cs, ce, color in h_segs:
            if r == top_row: tc = color
            if r == bot_row: bc = color
        for c, rs, re, color in v_segs:
            if c == left_col: lc = color
            if c == right_col: rc = color
        return tc, bc, lc, rc
    
    # Identify frame and pattern
    frame_comp = None
    pattern_comp = None
    
    if len(components) == 1:
        frame_comp = components[0]
        pattern_comp = None
    elif len(components) >= 2:
        best_score = -1
        best_assignment = None
        
        for i in range(len(components)):
            cand_frame = components[i]
            cand_pattern_cells = []
            for j in range(len(components)):
                if j != i:
                    cand_pattern_cells.extend(components[j])
            
            h_segs, v_segs = get_line_segments(cand_frame)
            rect = compute_rectangle(h_segs, v_segs)
            if rect is None:
                continue
            
            top_r, bot_r, left_c, right_c = rect
            int_h = bot_r - top_r - 1
            int_w = right_c - left_c - 1
            
            if int_h <= 0 or int_w <= 0:
                continue
            
            # Check if pattern bbox matches interior
            if cand_pattern_cells:
                pat_min_r = min(r for r, c in cand_pattern_cells)
                pat_max_r = max(r for r, c in cand_pattern_cells)
                pat_min_c = min(c for r, c in cand_pattern_cells)
                pat_max_c = max(c for r, c in cand_pattern_cells)
                pat_h = pat_max_r - pat_min_r + 1
                pat_w = pat_max_c - pat_min_c + 1
                
                # Score: how well does the pattern fit the interior?
                if pat_h == int_h and pat_w == int_w:
                    score = 100  # Perfect match
                else:
                    score = 0
            else:
                score = 50  # No pattern needed
            
            # Additional score for more colors in frame
            frame_colors = len(set(nonbg[p] for p in cand_frame))
            score += frame_colors
            
            if score > best_score:
                best_score = score
                best_assignment = (i, cand_frame, cand_pattern_cells, h_segs, v_segs, rect)
        
        if best_assignment:
            _, frame_comp, pattern_cells, _, _, _ = best_assignment
            pattern_comp = pattern_cells if pattern_cells else None
        else:
            frame_comp = components[0]
            pattern_comp = []
            for j in range(1, len(components)):
                pattern_comp.extend(components[j])
    
    # Compute frame rectangle
    h_segs, v_segs = get_line_segments(frame_comp)
    rect = compute_rectangle(h_segs, v_segs)
    if rect is None:
        return grid
    
    top_row, bot_row, left_col, right_col = rect
    out_rows = bot_row - top_row + 1
    out_cols = right_col - left_col + 1
    int_h = out_rows - 2
    int_w = out_cols - 2
    
    top_color, bot_color, left_color, right_color = get_border_colors(h_segs, v_segs, rect)
    
    # Build output
    output = [[bg] * out_cols for _ in range(out_rows)]
    
    # Place borders
    for oc in range(1, out_cols - 1):
        if top_color != bg: output[0][oc] = top_color
        if bot_color != bg: output[out_rows - 1][oc] = bot_color
    for or_ in range(1, out_rows - 1):
        if left_color != bg: output[or_][0] = left_color
        if right_color != bg: output[or_][out_cols - 1] = right_color
    
    if pattern_comp and len(pattern_comp) > 0:
        # Extract pattern shape relative to bounding box
        pat_min_r = min(r for r, c in pattern_comp)
        pat_min_c = min(c for r, c in pattern_comp)
        pat_max_r = max(r for r, c in pattern_comp)
        pat_max_c = max(c for r, c in pattern_comp)
        pat_h = pat_max_r - pat_min_r + 1
        pat_w = pat_max_c - pat_min_c + 1
        
        pattern = [[bg] * pat_w for _ in range(pat_h)]
        for r, c in pattern_comp:
            pattern[r - pat_min_r][c - pat_min_c] = nonbg[(r, c)]
        
        # Place pattern in interior
        for ir in range(int_h):
            for ic in range(int_w):
                pr, pc = ir, ic
                if pr < pat_h and pc < pat_w:
                    pat_val = pattern[pr][pc]
                else:
                    pat_val = bg
                
                if pat_val == bg:
                    continue
                
                r = ir + 1
                c = ic + 1
                d_top = r
                d_bot = out_rows - 1 - r
                d_left = c
                d_right = out_cols - 1 - c
                min_d = min(d_top, d_bot, d_left, d_right)
                
                nearest = []
                if d_top == min_d: nearest.append(top_color)
                if d_bot == min_d: nearest.append(bot_color)
                if d_left == min_d: nearest.append(left_color)
                if d_right == min_d: nearest.append(right_color)
                
                if len(nearest) == 1:
                    output[r][c] = nearest[0]
                else:
                    output[r][c] = pat_val
    else:
        # No pattern: project center cells of qualifying borders
        # A border qualifies if both perpendicular borders are non-bg
        if top_color != bg and left_color != bg and right_color != bg:
            output[1][out_cols // 2] = top_color
        if bot_color != bg and left_color != bg and right_color != bg:
            output[out_rows - 2][out_cols // 2] = bot_color
        if left_color != bg and top_color != bg and bot_color != bg:
            output[out_rows // 2][1] = left_color
        if right_color != bg and top_color != bg and bot_color != bg:
            output[out_rows // 2][out_cols - 2] = right_color
    
    return output
