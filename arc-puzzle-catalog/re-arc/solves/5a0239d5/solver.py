from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find separator rows (all same value)
    sep_rows = set()
    sep_val = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_rows.add(r)
            sep_val = grid[r][0]
    
    # Find separator columns
    sep_cols = set()
    for c in range(cols):
        if all(grid[r][c] == sep_val for r in range(rows)):
            sep_cols.add(c)
    
    # Find background value
    cnt = Counter()
    for r in range(rows):
        if r in sep_rows:
            continue
        for c in range(cols):
            if c in sep_cols:
                continue
            cnt[grid[r][c]] += 1
    bg = cnt.most_common(1)[0][0]
    
    # Find row bands and column bands
    row_bands = []
    start = None
    for r in range(rows):
        if r in sep_rows:
            if start is not None:
                row_bands.append((start, r - 1))
                start = None
        else:
            if start is None:
                start = r
    if start is not None:
        row_bands.append((start, rows - 1))
    
    col_bands = []
    start = None
    for c in range(cols):
        if c in sep_cols:
            if start is not None:
                col_bands.append((start, c - 1))
                start = None
        else:
            if start is None:
                start = c
    if start is not None:
        col_bands.append((start, cols - 1))
    
    output = [row[:] for row in grid]
    
    fill_map = {3: 2, 5: 0, 9: 6}
    
    for rb in row_bands:
        for cb in col_bands:
            marker = None
            for r in range(rb[0], rb[1] + 1):
                for c in range(cb[0], cb[1] + 1):
                    v = grid[r][c]
                    if v != bg and v != sep_val:
                        marker = v
                        break
                if marker is not None:
                    break
            
            fill_val = fill_map.get(marker, 8)
            
            for r in range(rb[0], rb[1] + 1):
                for c in range(cb[0], cb[1] + 1):
                    output[r][c] = fill_val
    
    return output
