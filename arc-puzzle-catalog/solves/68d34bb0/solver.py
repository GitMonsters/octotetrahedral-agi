def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    counter = Counter()
    for r in grid:
        counter.update(r)
    bg = counter.most_common(1)[0][0]
    
    # Find bar color: the color of full rows
    bar_color = None
    for r in range(rows):
        row_vals = set(grid[r][c] for c in range(cols))
        if len(row_vals) == 1 and grid[r][0] != bg:
            bar_color = grid[r][0]
            break
    if bar_color is None:
        return grid
    
    bar_rows = [r for r in range(rows) if all(grid[r][c] == bar_color for c in range(cols))]
    bar_cols = [c for c in range(cols) if all(grid[r][c] == bar_color for r in range(rows))]
    bar_row_set = set(bar_rows)
    bar_col_set = set(bar_cols)
    
    row_sections = []
    prev = -1
    for br in bar_rows:
        if br > prev + 1:
            row_sections.append(list(range(prev + 1, br)))
        prev = br
    if prev < rows - 1:
        row_sections.append(list(range(prev + 1, rows)))
    
    col_sections = []
    prev = -1
    for bc in bar_cols:
        if bc > prev + 1:
            col_sections.append(list(range(prev + 1, bc)))
        prev = bc
    if prev < cols - 1:
        col_sections.append(list(range(prev + 1, cols)))
    
    def get_grid_coord(r, c):
        for i, rs in enumerate(row_sections):
            if r in rs:
                for j, cs in enumerate(col_sections):
                    if c in cs:
                        return (i, j, rs.index(r), cs.index(c))
        return None
    
    # Find all marker cells (non-bg, NOT on bar rows/cols)
    markers = {}
    for r in range(rows):
        if r in bar_row_set:
            continue
        for c in range(cols):
            if c in bar_col_set:
                continue
            v = grid[r][c]
            if v != bg:
                coord = get_grid_coord(r, c)
                if coord:
                    rs, cs, rw, cw = coord
                    markers.setdefault(v, []).append((rs, cs, rw, cw))
    
    out = [row[:] for row in grid]
    
    for color, positions in markers.items():
        rw = positions[0][2]
        cw = positions[0][3]
        
        by_row = {}
        for rs, cs, _, _ in positions:
            by_row.setdefault(rs, set()).add(cs)
        
        for rs, cs_set in by_row.items():
            if len(cs_set) >= 2:
                for cs in range(min(cs_set), max(cs_set) + 1):
                    actual_r = row_sections[rs][rw]
                    actual_c = col_sections[cs][cw]
                    out[actual_r][actual_c] = color
        
        by_col = {}
        for rs, cs, _, _ in positions:
            by_col.setdefault(cs, set()).add(rs)
        
        for cs, rs_set in by_col.items():
            if len(rs_set) >= 2:
                for rs in range(min(rs_set), max(rs_set) + 1):
                    actual_r = row_sections[rs][rw]
                    actual_c = col_sections[cs][cw]
                    out[actual_r][actual_c] = color
    
    return out
