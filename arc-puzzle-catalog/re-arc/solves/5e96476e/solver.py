def transform(grid):
    """
    Fill interior gaps within rectangular regions formed by lines.
    
    Algorithm:
    1. Identify the line color (minority non-background value)
    2. Find all horizontal and vertical line segments
    3. Locate rectangles formed by pairs of horizontal and vertical lines
    4. For each rectangle, fill background cells that are enclosed by lines
       (have lines in all 4 directions: left, right, up, down)
    
    The key insight is that gaps/holes completely surrounded by line segments
    within a rectangular boundary should be filled with color 3.
    """
    import numpy as np
    
    grid = np.array(grid, dtype=int)
    output = grid.copy()
    h, w = grid.shape
    
    # Identify background and line colors
    unique_vals = np.unique(grid)
    if len(unique_vals) == 1:
        return grid.tolist()
    
    val_counts = {v: np.sum(grid == v) for v in unique_vals}
    line_color = min(val_counts, key=val_counts.get)
    background = max(val_counts, key=val_counts.get)
    
    # Extract horizontal line segments
    h_lines = {}  # row -> list of (col_start, col_end)
    for r in range(h):
        start = None
        for c in range(w):
            if grid[r, c] == line_color:
                if start is None:
                    start = c
            else:
                if start is not None:
                    if r not in h_lines:
                        h_lines[r] = []
                    h_lines[r].append((start, c - 1))
                    start = None
        if start is not None:
            if r not in h_lines:
                h_lines[r] = []
            h_lines[r].append((start, w - 1))
    
    # Extract vertical line segments
    v_lines = {}  # col -> list of (row_start, row_end)
    for c in range(w):
        start = None
        for r in range(h):
            if grid[r, c] == line_color:
                if start is None:
                    start = r
            else:
                if start is not None:
                    if c not in v_lines:
                        v_lines[c] = []
                    v_lines[c].append((start, r - 1))
                    start = None
        if start is not None:
            if c not in v_lines:
                v_lines[c] = []
            v_lines[c].append((start, h - 1))
    
    # Find rectangles by looking for pairs of horizontal and vertical lines
    rectangles = []
    h_line_rows = sorted(h_lines.keys())
    v_line_cols = sorted(v_lines.keys())
    
    for r_idx in range(len(h_line_rows)):
        for r2_idx in range(r_idx + 1, len(h_line_rows)):
            r1, r2 = h_line_rows[r_idx], h_line_rows[r2_idx]
            h1_ranges = h_lines[r1]
            h2_ranges = h_lines[r2]
            
            for c_idx in range(len(v_line_cols)):
                for c2_idx in range(c_idx + 1, len(v_line_cols)):
                    c1, c2 = v_line_cols[c_idx], v_line_cols[c2_idx]
                    v1_ranges = v_lines[c1]
                    v2_ranges = v_lines[c2]
                    
                    # Check if these form a complete rectangle
                    h1_valid = any(s <= c1 and e >= c2 for s, e in h1_ranges)
                    h2_valid = any(s <= c1 and e >= c2 for s, e in h2_ranges)
                    v1_valid = any(s <= r1 and e >= r2 for s, e in v1_ranges)
                    v2_valid = any(s <= r1 and e >= r2 for s, e in v2_ranges)
                    
                    if h1_valid and h2_valid and v1_valid and v2_valid:
                        rectangles.append((r1, c1, r2, c2))
    
    # For each rectangle, fill interior gaps that are enclosed by lines
    for r1, c1, r2, c2 in rectangles:
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if output[r, c] == background:
                    # Check if this cell is surrounded by lines
                    has_line_left = any(grid[r, cc] == line_color for cc in range(c1, c))
                    has_line_right = any(grid[r, cc] == line_color for cc in range(c + 1, c2 + 1))
                    has_line_up = any(grid[rr, c] == line_color for rr in range(r1, r))
                    has_line_down = any(grid[rr, c] == line_color for rr in range(r + 1, r2 + 1))
                    
                    if has_line_left and has_line_right and has_line_up and has_line_down:
                        output[r, c] = 3
    
    return output.tolist()
