def transform(grid):
    H, W = len(grid), len(grid[0])
    sep_color = None
    sep_start = sep_end = -1
    sep_dir = None
    
    for r in range(H):
        if len(set(grid[r])) == 1:
            c = grid[r][0]
            start = r
            while start > 0 and len(set(grid[start-1])) == 1 and grid[start-1][0] == c:
                start -= 1
            end = r
            while end < H-1 and len(set(grid[end+1])) == 1 and grid[end+1][0] == c:
                end += 1
            if end - start + 1 > (sep_end - sep_start + 1 if sep_start >= 0 else 0):
                sep_color, sep_start, sep_end, sep_dir = c, start, end, 'h'
    
    for c_idx in range(W):
        col = [grid[r][c_idx] for r in range(H)]
        if len(set(col)) == 1:
            c = col[0]
            start = c_idx
            while start > 0 and len(set(grid[r][start-1] for r in range(H))) == 1 and grid[0][start-1] == c:
                start -= 1
            end = c_idx
            while end < W-1 and len(set(grid[r][end+1] for r in range(H))) == 1 and grid[0][end+1] == c:
                end += 1
            if end - start + 1 > (sep_end - sep_start + 1 if sep_start >= 0 else 0):
                sep_color, sep_start, sep_end, sep_dir = c, start, end, 'v'
    
    output = [row[:] for row in grid]
    
    if sep_dir == 'h':
        half_colors = {}
        for r in list(range(0, sep_start)) + list(range(sep_end+1, H)):
            for c in range(W):
                v = grid[r][c]
                if v != sep_color:
                    half_colors[v] = half_colors.get(v, 0) + 1
        half_color = max(half_colors, key=half_colors.get) if half_colors else sep_color
        if sep_start > 0:
            for c in range(W):
                col_vals = [grid[r][c] for r in range(sep_start)]
                count = sum(1 for v in col_vals if v == half_color)
                new_col = [half_color] * count + [sep_color] * (sep_start - count)
                for r in range(sep_start):
                    output[r][c] = new_col[r]
        if sep_end < H - 1:
            for c in range(W):
                col_vals = [grid[r][c] for r in range(sep_end+1, H)]
                count = sum(1 for v in col_vals if v == half_color)
                n = H - 1 - sep_end
                new_col = [sep_color] * (n - count) + [half_color] * count
                for idx, r in enumerate(range(sep_end+1, H)):
                    output[r][c] = new_col[idx]
    elif sep_dir == 'v':
        half_colors = {}
        for r in range(H):
            for c in list(range(0, sep_start)) + list(range(sep_end+1, W)):
                v = grid[r][c]
                if v != sep_color:
                    half_colors[v] = half_colors.get(v, 0) + 1
        half_color = max(half_colors, key=half_colors.get) if half_colors else sep_color
        if sep_start > 0:
            for r in range(H):
                row_vals = [grid[r][c] for c in range(sep_start)]
                count = sum(1 for v in row_vals if v == half_color)
                new_row = [half_color] * count + [sep_color] * (sep_start - count)
                for c in range(sep_start):
                    output[r][c] = new_row[c]
        if sep_end < W - 1:
            for r in range(H):
                row_vals = [grid[r][c] for c in range(sep_end+1, W)]
                count = sum(1 for v in row_vals if v == half_color)
                n = W - 1 - sep_end
                new_row = [sep_color] * (n - count) + [half_color] * count
                for idx, c in enumerate(range(sep_end+1, W)):
                    output[r][c] = new_row[idx]
    return output
