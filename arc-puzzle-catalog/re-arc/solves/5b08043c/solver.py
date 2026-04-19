from collections import Counter, defaultdict

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    cnt = Counter()
    for r in range(rows):
        for c in range(cols):
            cnt[grid[r][c]] += 1
    bg = cnt.most_common(1)[0][0]
    
    # Template row from bottom, template col from right
    trow = grid[-1][:]
    tcol = [grid[r][-1] for r in range(rows)]
    
    # V-lines: non-bg in template row
    v_lines = {}
    for c in range(cols):
        if trow[c] != bg:
            v_lines[c] = trow[c]
    
    # H-lines: non-bg in template col
    h_lines = {}
    for r in range(rows):
        if tcol[r] != bg:
            h_lines[r] = tcol[r]
    
    # Virtual V-columns: midpoint of same-color V-line pairs
    color_to_cols = defaultdict(list)
    for c, color in v_lines.items():
        color_to_cols[color].append(sorted_c := c)
    # rebuild properly
    color_to_cols = defaultdict(list)
    for c in sorted(v_lines.keys()):
        color_to_cols[v_lines[c]].append(c)
    
    virtual_v = set()
    for color, col_list in color_to_cols.items():
        if len(col_list) == 2:
            mid = (col_list[0] + col_list[1]) // 2
            if mid not in v_lines:
                virtual_v.add(mid)
    
    all_v = set(v_lines.keys()) | virtual_v
    
    output = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            is_h = r in h_lines
            is_v = c in all_v
            if is_h and is_v:
                output[r][c] = 8
            elif is_h:
                output[r][c] = h_lines[r]
            elif c in v_lines:
                output[r][c] = v_lines[c]
            else:
                output[r][c] = bg
    
    return output
