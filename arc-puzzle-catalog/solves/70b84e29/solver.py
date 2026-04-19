def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find non-bg color
    non_bg_colors = [c for c in set(flat) if c != bg]
    if not non_bg_colors:
        return grid
    non_bg = non_bg_colors[0]
    
    # Find longest consecutive runs in each row
    row_runs = []
    for r in range(rows):
        best_start, best_len = 0, 0
        start, length = 0, 0
        for c in range(cols):
            if grid[r][c] == non_bg:
                if length == 0: start = c
                length += 1
            else:
                if length > best_len:
                    best_start, best_len = start, length
                length = 0
        if length > best_len:
            best_start, best_len = start, length
        row_runs.append((best_len, best_start, best_start + best_len - 1, r))
    
    # Find longest consecutive runs in each column
    col_runs = []
    for c in range(cols):
        best_start, best_len = 0, 0
        start, length = 0, 0
        for r in range(rows):
            if grid[r][c] == non_bg:
                if length == 0: start = r
                length += 1
            else:
                if length > best_len:
                    best_start, best_len = start, length
                length = 0
        if length > best_len:
            best_start, best_len = start, length
        col_runs.append((best_len, best_start, best_start + best_len - 1, c))
    
    row_runs.sort(reverse=True)
    col_runs.sort(reverse=True)
    
    max_row_run = row_runs[0][0] if row_runs else 0
    max_col_run = col_runs[0][0] if col_runs else 0
    
    if max_row_run >= max_col_run:
        # Two horizontal border lines
        line1 = row_runs[0]
        line2 = row_runs[1]
        col_start = max(line1[1], line2[1])
        col_end = min(line1[2], line2[2])
        row_start = min(line1[3], line2[3])
        row_end = max(line1[3], line2[3])
    else:
        # Two vertical border lines
        line1 = col_runs[0]
        line2 = col_runs[1]
        row_start = max(line1[1], line2[1])
        row_end = min(line1[2], line2[2])
        col_start = min(line1[3], line2[3])
        col_end = max(line1[3], line2[3])
    
    rect_h = row_end - row_start + 1
    rect_w = col_end - col_start + 1
    side = max(rect_h, rect_w)
    pad_top = (side - rect_h) // 2
    pad_left = (side - rect_w) // 2
    
    output = [[bg] * side for _ in range(side)]
    for r in range(rect_h):
        for c in range(rect_w):
            output[r + pad_top][c + pad_left] = grid[row_start + r][col_start + c]
    
    return output
