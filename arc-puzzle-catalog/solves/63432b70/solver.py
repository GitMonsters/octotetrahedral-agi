from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    non_bg = Counter(v for r in grid for v in r if v != bg)
    sorted_colors = non_bg.most_common()
    inter_color = sorted_colors[1][0] if len(sorted_colors) >= 2 else sorted_colors[0][0]
    
    # For each non-bg pixel, find horizontal and vertical run lengths
    def h_run_len(r, c):
        color = grid[r][c]
        l = c
        while l > 0 and grid[r][l-1] == color: l -= 1
        ri = c
        while ri < cols-1 and grid[r][ri+1] == color: ri += 1
        return ri - l + 1
    
    def v_run_len(r, c):
        color = grid[r][c]
        t = r
        while t > 0 and grid[t-1][c] == color: t -= 1
        b = r
        while b < rows-1 and grid[b+1][c] == color: b += 1
        return b - t + 1
    
    # Determine row-lines and col-lines
    row_lines = {}
    col_lines = {}
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg:
                continue
            color = grid[r][c]
            h = h_run_len(r, c)
            v = v_run_len(r, c)
            
            if h >= 2 and h > v:
                # Horizontal pixel -> defines row-line
                if r not in row_lines:
                    row_lines[r] = color
            elif v >= 2 and v > h:
                # Vertical pixel -> defines col-line
                if c not in col_lines:
                    col_lines[c] = color
            elif h >= 2 and v >= 2:
                # Tie -> prefer vertical
                if c not in col_lines:
                    col_lines[c] = color
            # Single isolated pixel: ignore (length 1 in both dirs)
    
    # Build output
    out = [[bg]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            r_line = r in row_lines
            c_line = c in col_lines
            if r_line and c_line:
                out[r][c] = inter_color
            elif r_line:
                out[r][c] = row_lines[r]
            elif c_line:
                out[r][c] = col_lines[c]
    
    return out
