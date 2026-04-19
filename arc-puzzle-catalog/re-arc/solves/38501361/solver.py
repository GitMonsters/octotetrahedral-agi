def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    bg = grid[0, 0]
    
    # Find horizontal lines (length >= 3, same color)
    def find_horiz_lines():
        lines = []
        for r in range(h):
            c = 0
            while c < w:
                if grid[r, c] != bg:
                    color = grid[r, c]
                    start = c
                    while c < w and grid[r, c] == color:
                        c += 1
                    if c - start >= 4:  # Line of length >= 4
                        lines.append((r, start, c - 1, color))
                else:
                    c += 1
        return lines
    
    # Find vertical lines (length >= 3, same color)
    def find_vert_lines():
        lines = []
        for c in range(w):
            r = 0
            while r < h:
                if grid[r, c] != bg:
                    color = grid[r, c]
                    start = r
                    while r < h and grid[r, c] == color:
                        r += 1
                    if r - start >= 4:
                        lines.append((start, r - 1, c, color))
                else:
                    r += 1
        return lines
    
    h_lines = find_horiz_lines()
    v_lines = find_vert_lines()
    
    # Find frame configuration
    # Top/bottom must be horizontal, left/right must be vertical
    best_frame = None
    
    for top in h_lines:
        for bot in h_lines:
            t_r, t_c1, t_c2, t_col = top
            b_r, b_c1, b_c2, b_col = bot
            if t_r >= b_r:
                continue
            if t_c1 != b_c1 or t_c2 != b_c2:
                continue
                
            # Look for vertical lines adjacent to horizontal lines
            for left in v_lines:
                for right in v_lines:
                    l_r1, l_r2, l_c, l_col = left
                    r_r1, r_r2, r_c, r_col = right
                    if l_c >= r_c:
                        continue
                    
                    # Check that vertical lines are just outside horizontal lines
                    if l_c != t_c1 - 1:
                        continue
                    if r_c != t_c2 + 1:
                        continue
                    # Vertical lines span from row after top to row before bottom
                    if l_r1 != t_r + 1 or l_r2 != b_r - 1:
                        continue
                    if r_r1 != t_r + 1 or r_r2 != b_r - 1:
                        continue
                    
                    best_frame = (t_r, b_r, l_c, r_c, t_col, b_col, l_col, r_col)
    
    if best_frame:
        top_r, bot_r, left_c, right_c, t_col, b_col, l_col, r_col = best_frame
        
        # Collect frame pixels
        frame_pixels = set()
        for c in range(left_c + 1, right_c):
            frame_pixels.add((top_r, c))
            frame_pixels.add((bot_r, c))
        for r in range(top_r + 1, bot_r):
            frame_pixels.add((r, left_c))
            frame_pixels.add((r, right_c))
        
        # Find external pattern
        pattern_pixels = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg and (r, c) not in frame_pixels:
                    pattern_pixels.append((r, c, grid[r, c]))
        
        out_h = bot_r - top_r + 1
        out_w = right_c - left_c + 1
        output = np.full((out_h, out_w), bg, dtype=grid.dtype)
        
        # Draw frame
        for c in range(1, out_w - 1):
            output[0, c] = t_col
            output[out_h - 1, c] = b_col
        for r in range(1, out_h - 1):
            output[r, 0] = l_col
            output[r, out_w - 1] = r_col
        
        # Map pattern based on its relation to frame
        if pattern_pixels:
            p_rows = [p[0] for p in pattern_pixels]
            p_cols = [p[1] for p in pattern_pixels]
            p_min_r, p_max_r = min(p_rows), max(p_rows)
            p_min_c, p_max_c = min(p_cols), max(p_cols)
            
            inner_h = out_h - 2  
            inner_w = out_w - 2
            
            # Pattern is usually at a corner - figure out which
            center_r = (top_r + bot_r) / 2
            center_c = (left_c + right_c) / 2
            pat_center_r = (p_min_r + p_max_r) / 2
            pat_center_c = (p_min_c + p_max_c) / 2
            
            # Check if pattern is to bottom-left of frame
            if pat_center_r > center_r and pat_center_c < center_c:
                # Map bottom-left pattern to top-right area in output
                # Flip diagonally
                for pr, pc, pv in pattern_pixels:
                    # Relative position in pattern
                    rel_r = pr - p_min_r
                    rel_c = pc - p_min_c
                    # Place in output (flip from bottom-left to top-right)
                    nr = 1 + (inner_h - 1 - (p_max_r - p_min_r) + rel_r)
                    nc = out_w - 2 - rel_c
                    if 1 <= nr < out_h - 1 and 1 <= nc < out_w - 1:
                        output[nr, nc] = pv
            else:
                for pr, pc, pv in pattern_pixels:
                    nr = 1 + (pr - p_min_r)
                    nc = 1 + (pc - p_min_c)
                    if 0 <= nr < out_h and 0 <= nc < out_w:
                        output[nr, nc] = pv
        
        return output.tolist()
    
    # Fallback: just 2 lines (top and bottom) - Train 2 case
    if len(h_lines) >= 2:
        # Find two horizontal lines
        h_lines_sorted = sorted(h_lines, key=lambda x: x[0])
        top = h_lines_sorted[0]
        bot = h_lines_sorted[-1]
        t_r, t_c1, t_c2, t_col = top
        b_r, b_c1, b_c2, b_col = bot
        
        # Output height = bottom - top + 1, width = line length + 2 (for corners)
        line_len = t_c2 - t_c1 + 1
        out_h = b_r - t_r + 1
        out_w = line_len + 2
        
        output = np.full((out_h, out_w), bg, dtype=grid.dtype)
        
        # Draw top and bottom edges
        for c in range(1, out_w - 1):
            output[0, c] = t_col
            output[out_h - 1, c] = b_col
        
        # Find pattern between the lines
        pattern_pixels = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg:
                    if r != t_r and r != b_r:  # Not on the lines
                        pattern_pixels.append((r, c, grid[r, c]))
        
        if pattern_pixels:
            p_rows = [p[0] for p in pattern_pixels]
            p_cols = [p[1] for p in pattern_pixels]
            p_min_r = min(p_rows)
            p_min_c = min(p_cols)
            
            for pr, pc, pv in pattern_pixels:
                nr = pr - t_r
                nc = 1 + (pc - t_c1)
                if 0 <= nr < out_h and 0 <= nc < out_w:
                    output[nr, nc] = pv
        
        return output.tolist()
    
    return grid.tolist()

if __name__ == "__main__":
    import json
    data = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))
    puzzle = data['38501361']
    
    all_pass = True
    for i, ex in enumerate(puzzle['train']):
        result = transform(ex['input'])
        expected = ex['output']
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            print(f"Train {i}: FAIL")
            print(f"Exp: {expected}")
            print(f"Got: {result}")
            all_pass = False
    print(f"\nAll pass: {all_pass}")
