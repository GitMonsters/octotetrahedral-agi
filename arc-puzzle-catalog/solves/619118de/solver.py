def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find separator row or column
    sep_row = sep_col = None
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            sep_row = r
            break
    if sep_row is None:
        for c in range(cols):
            vals = set(grid[r][c] for r in range(rows))
            if len(vals) == 1 and grid[0][c] != bg:
                sep_col = c
                break

    out = [row[:] for row in grid]

    if sep_row is not None:
        half1 = list(range(0, sep_row))
        half2 = list(range(sep_row + 1, rows))
        # Pattern side = side with more distinct non-bg colors
        def distinct(rng):
            return len(set(grid[r][c] for r in rng for c in range(cols)) - {bg})
        if distinct(half1) >= distinct(half2):
            pat_rows, can_rows = half1, half2
        else:
            pat_rows, can_rows = half2, half1

        # Extract pattern bounding box
        ph = len(pat_rows)
        non_bg = [(r, c) for r in pat_rows for c in range(cols) if grid[r][c] != bg]
        min_c = min(c for _, c in non_bg)
        max_c = max(c for _, c in non_bg)
        pw = max_c - min_c + 1
        if pw < ph:
            if min_c + ph - 1 < cols:
                max_c = min_c + ph - 1
            else:
                min_c = max_c - ph + 1
            pw = ph

        pattern = [[grid[r][c] for c in range(min_c, min_c + pw)] for r in pat_rows]

        # Find dots on canvas
        dots = [(r, c) for r in can_rows for c in range(cols) if grid[r][c] != bg]

        # Stamp centered on each dot
        cr, cc = ph // 2, pw // 2
        for dr, dc in dots:
            for pr in range(ph):
                for pc in range(pw):
                    tr, tc = dr - cr + pr, dc - cc + pc
                    if 0 <= tr < rows and 0 <= tc < cols:
                        out[tr][tc] = pattern[pr][pc]

    elif sep_col is not None:
        half1 = list(range(0, sep_col))
        half2 = list(range(sep_col + 1, cols))
        def distinct(crng):
            return len(set(grid[r][c] for r in range(rows) for c in crng) - {bg})
        if distinct(half1) >= distinct(half2):
            pat_cols, can_cols = half1, half2
        else:
            pat_cols, can_cols = half2, half1

        pw = len(pat_cols)
        non_bg = [(r, c) for r in range(rows) for c in pat_cols if grid[r][c] != bg]
        min_r = min(r for r, _ in non_bg)
        max_r = max(r for r, _ in non_bg)
        ph = max_r - min_r + 1
        if ph < pw:
            if min_r + pw - 1 < rows:
                max_r = min_r + pw - 1
            else:
                min_r = max_r - pw + 1
            ph = pw

        pattern = [[grid[r][c] for c in pat_cols] for r in range(min_r, min_r + ph)]

        dots = [(r, c) for r in range(rows) for c in can_cols if grid[r][c] != bg]

        cr, cc = ph // 2, pw // 2
        for dr, dc in dots:
            for pr in range(ph):
                for pc in range(pw):
                    tr, tc = dr - cr + pr, dc - cc + pc
                    if 0 <= tr < rows and 0 <= tc < cols:
                        out[tr][tc] = pattern[pr][pc]

    return out
