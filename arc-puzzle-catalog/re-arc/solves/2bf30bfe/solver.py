import copy
from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find background color (most common)
    flat = [input_grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find the rectangle outline color and its bounding box
    non_bg_colors = set(flat) - {bg}
    rect_color = None
    rect = None
    
    for color in non_bg_colors:
        positions = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == color]
        if not positions:
            continue
        r_min = min(r for r, c in positions)
        r_max = max(r for r, c in positions)
        c_min = min(c for r, c in positions)
        c_max = max(c for r, c in positions)
        
        top_ok = all(input_grid[r_min][c] == color for c in range(c_min, c_max+1))
        bot_ok = all(input_grid[r_max][c] == color for c in range(c_min, c_max+1))
        left_ok = all(input_grid[r][c_min] == color for r in range(r_min, r_max+1))
        right_ok = all(input_grid[r][c_max] == color for r in range(r_min, r_max+1))
        
        if top_ok and bot_ok and left_ok and right_ok:
            rect_color = color
            rect = (r_min, r_max, c_min, c_max)
            break
    
    if rect_color is None:
        # Fallback: try with bg as 0
        for color in set(flat) - {0}:
            positions = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == color]
            if not positions:
                continue
            r_min = min(r for r, c in positions)
            r_max = max(r for r, c in positions)
            c_min = min(c for r, c in positions)
            c_max = max(c for r, c in positions)
            top_ok = all(input_grid[r_min][c] == color for c in range(c_min, c_max+1))
            bot_ok = all(input_grid[r_max][c] == color for c in range(c_min, c_max+1))
            left_ok = all(input_grid[r][c_min] == color for r in range(r_min, r_max+1))
            right_ok = all(input_grid[r][c_max] == color for r in range(r_min, r_max+1))
            if top_ok and bot_ok and left_ok and right_ok:
                rect_color = color
                rect = (r_min, r_max, c_min, c_max)
                bg = 0  # override
                break
    
    r1, r2, c1, c2 = rect
    
    # Identify hole color (non-rect color inside the rectangle)
    hole_color = None
    for r in range(r1+1, r2):
        for c in range(c1+1, c2):
            if input_grid[r][c] != rect_color:
                hole_color = input_grid[r][c]
                break
        if hole_color is not None:
            break
    if hole_color is None:
        hole_color = 0  # default
    
    # First/last inner rows and cols
    fi_r = r1 + 1
    li_r = r2 - 1
    fi_c = c1 + 1
    li_c = c2 - 1
    
    # Deep inside range
    di_r_start = fi_r + 1
    di_r_end = li_r - 1
    di_c_start = fi_c + 1
    di_c_end = li_c - 1
    
    # Projecting columns: holes in first/last inner row (deep inside cols only)
    proj_cols = set()
    for c in range(di_c_start, di_c_end + 1):
        if input_grid[fi_r][c] == hole_color:
            proj_cols.add(c)
        if input_grid[li_r][c] == hole_color:
            proj_cols.add(c)
    
    # Projecting rows: holes in first/last inner col (deep inside rows only)
    proj_rows = set()
    for r in range(di_r_start, di_r_end + 1):
        if input_grid[r][fi_c] == hole_color:
            proj_rows.add(r)
        if input_grid[r][li_c] == hole_color:
            proj_rows.add(r)
    
    # Build output
    output = copy.deepcopy(input_grid)
    
    # Fill entire rectangle interior with rect_color
    for r in range(fi_r, li_r + 1):
        for c in range(fi_c, li_c + 1):
            output[r][c] = rect_color
    
    # Restore holes in deep inside where NOT projecting
    for r in range(di_r_start, di_r_end + 1):
        for c in range(di_c_start, di_c_end + 1):
            if r not in proj_rows and c not in proj_cols:
                if input_grid[r][c] == hole_color:
                    output[r][c] = hole_color
    
    # Project outside: projecting columns above and below
    for c in proj_cols:
        for r in range(0, r1):
            if output[r][c] == bg:
                output[r][c] = hole_color
        for r in range(r2 + 1, rows):
            if output[r][c] == bg:
                output[r][c] = hole_color
    
    # Project outside: projecting rows left and right
    for r in proj_rows:
        for c in range(0, c1):
            if output[r][c] == bg:
                output[r][c] = hole_color
        for c in range(c2 + 1, cols):
            if output[r][c] == bg:
                output[r][c] = hole_color
    
    return output
