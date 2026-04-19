def transform(grid):
    R, C = len(grid), len(grid[0])
    from collections import Counter
    
    div_rows = []
    div_color = None
    for r in range(R):
        if len(set(grid[r])) == 1:
            div_rows.append(r)
            div_color = grid[r][0]
    
    div_cols = []
    for c in range(C):
        if all(grid[r][c] == div_color for r in range(R)):
            div_cols.append(c)
    
    non_div = []
    for r in range(R):
        if r in div_rows: continue
        for c in range(C):
            if c in div_cols: continue
            non_div.append(grid[r][c])
    bg_color = Counter(non_div).most_common(1)[0][0]
    
    color_map = {1: 1, 2: 4, 3: 4, 6: 0}
    
    row_ranges = []
    prev = 0
    for dr in sorted(div_rows) + [R]:
        if dr > prev:
            row_ranges.append((prev, dr))
        prev = dr + 1
    
    col_ranges = []
    prev = 0
    for dc in sorted(div_cols) + [C]:
        if dc > prev:
            col_ranges.append((prev, dc))
        prev = dc + 1
    
    out = [row[:] for row in grid]
    
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            marker = None
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if grid[r][c] != bg_color:
                        marker = grid[r][c]
                        break
                if marker is not None:
                    break
            
            fill = color_map.get(marker, 4) if marker is not None else color_map.get(bg_color, bg_color)
            
            for r in range(r0, r1):
                for c in range(c0, c1):
                    out[r][c] = fill
    
    return out
