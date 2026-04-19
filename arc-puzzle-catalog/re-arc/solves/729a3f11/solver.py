from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    pixels = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                pixels[r] = (c, grid[r][c])
    
    color_rows = {}
    for r, (c, v) in pixels.items():
        color_rows.setdefault(v, []).append((r, c))
    
    result = [[bg]*cols for _ in range(rows)]
    
    for r, (c, v) in pixels.items():
        for i in range(c + 1):
            result[r][i] = v if (i % 2) == (c % 2) else 8
    
    for v, positions in color_rows.items():
        positions.sort()
        for idx in range(len(positions) - 1):
            r1, c1 = positions[idx]
            r2, c2 = positions[idx + 1]
            if c2 <= c1:
                continue
            empty_start = None
            empty_runs = []
            for r in range(r1 + 1, r2):
                if r not in pixels:
                    if empty_start is None:
                        empty_start = r
                else:
                    if empty_start is not None:
                        empty_runs.append((empty_start, r - 1))
                        empty_start = None
            if empty_start is not None:
                empty_runs.append((empty_start, r2 - 1))
            
            for run_start, run_end in empty_runs:
                run_len = run_end - run_start + 1
                if run_len < 4:
                    continue
                for r in range(run_start, run_end + 1):
                    dist_up = r - r1
                    dist_down = r2 - r
                    if dist_up > dist_down:
                        continue
                    col_end = c2 - dist_down
                    if col_end < 0:
                        continue
                    for i in range(min(col_end + 1, cols)):
                        if (i % 2) == (c2 % 2):
                            result[r][i] = 8
    
    return result
