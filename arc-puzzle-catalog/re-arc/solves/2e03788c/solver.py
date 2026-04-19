from collections import Counter

def transform(input_grid):
    inp = input_grid
    rows, cols = len(inp), len(inp[0])
    all_vals = Counter()
    for r in inp:
        all_vals.update(r)
    fill = all_vals.most_common(1)[0][0]
    
    sep_rows, sep_cols = [], []
    for r in range(rows):
        if sum(1 for v in inp[r] if v != fill) > cols * 0.5:
            sep_rows.append(r)
    for c in range(cols):
        if sum(1 for r in range(rows) if inp[r][c] != fill) > rows * 0.5:
            sep_cols.append(c)
    
    gridline_counts = Counter()
    sc_set = set(sep_cols)
    for r in sep_rows:
        for c in range(cols):
            if c not in sc_set:
                gridline_counts[inp[r][c]] += 1
    gridline = gridline_counts.most_common(1)[0][0]
    
    fill_at_intersection = any(inp[r][c] == fill for r in sep_rows for c in sep_cols)
    
    active_rows = list(sep_rows)
    while active_rows and all(inp[active_rows[0]][c] == gridline for c in sep_cols):
        active_rows.pop(0)
    while active_rows and all(inp[active_rows[-1]][c] == gridline for c in sep_cols):
        active_rows.pop()
    
    active_cols = list(sep_cols)
    while active_cols and all(inp[r][active_cols[0]] == gridline for r in sep_rows):
        active_cols.pop(0)
    while active_cols and all(inp[r][active_cols[-1]] == gridline for r in sep_rows):
        active_cols.pop()
    
    allgrid_rows = set(r for r in active_rows if all(inp[r][c] == gridline for c in active_cols))
    allgrid_cols = set(c for c in active_cols if all(inp[r][c] == gridline for r in active_rows))
    
    out_r = len(active_rows) - 1
    out_c = len(active_cols) - 1
    
    result = []
    for i in range(out_r):
        row = []
        for j in range(out_c):
            r1, r2 = active_rows[i], active_rows[i+1]
            c1, c2 = active_cols[j], active_cols[j+1]
            corners = [inp[r1][c1], inp[r1][c2], inp[r2][c1], inp[r2][c2]]
            
            has_allgrid_boundary = (r1 in allgrid_rows or r2 in allgrid_rows or
                                    c1 in allgrid_cols or c2 in allgrid_cols)
            
            if has_allgrid_boundary:
                row.append(fill)
            elif len(set(corners)) == 1:
                val = corners[0]
                if val == gridline and fill_at_intersection:
                    row.append(fill)
                else:
                    row.append(val)
            else:
                row.append(fill)
        result.append(row)
    return result
