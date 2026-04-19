def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    from collections import Counter
    
    # Find background color
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find complete rows and cols by color
    row_colors = {}
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1:
            row_colors.setdefault(grid[r][0], []).append(r)
    
    col_colors = {}
    for c in range(cols):
        vals = set(grid[r][c] for r in range(rows))
        if len(vals) == 1:
            col_colors.setdefault(grid[0][c], []).append(c)
    
    # Separator color forms both complete rows and complete columns
    sep_color = None
    cand_rows = []
    cand_cols = []
    for color in set(row_colors.keys()) & set(col_colors.keys()):
        if len(row_colors[color]) >= 1 and len(col_colors[color]) >= 1:
            sep_color = color
            cand_rows = sorted(row_colors[color])
            cand_cols = sorted(col_colors[color])
            break
    
    # Find regular separator pattern
    def find_regular_seps(candidates, total_size):
        """Find subset of candidates forming evenly-spaced separators."""
        cand_set = set(candidates)
        for s in range(1, total_size):
            seps = []
            pos = s
            while pos < total_size:
                seps.append(pos)
                pos += s + 1
            if all(sp in cand_set for sp in seps) and len(seps) >= 1:
                # Verify sections fill the grid
                last_end = seps[-1] + s
                if last_end == total_size - 1 or last_end == total_size:
                    return seps, s
        return candidates, None
    
    sep_rows, sec_h = find_regular_seps(cand_rows, rows)
    sep_cols, sec_w = find_regular_seps(cand_cols, cols)
    
    # Build section ranges
    def get_ranges(seps, total):
        ranges = []
        start = 0
        for sp in sorted(seps):
            if sp > start:
                ranges.append((start, sp - 1))
            start = sp + 1
        if start < total:
            ranges.append((start, total - 1))
        return ranges
    
    row_ranges = get_ranges(sep_rows, rows)
    col_ranges = get_ranges(sep_cols, cols)
    
    # Find the 0 cell
    zero_r = zero_c = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                zero_r, zero_c = r, c
                break
        if zero_r is not None:
            break
    
    # Find source section
    src_si = src_sj = None
    for si, (r1, r2) in enumerate(row_ranges):
        for sj, (c1, c2) in enumerate(col_ranges):
            if r1 <= zero_r <= r2 and c1 <= zero_c <= c2:
                src_si, src_sj = si, sj
                break
    
    src_r1, src_r2 = row_ranges[src_si]
    src_c1, src_c2 = col_ranges[src_sj]
    rel_r = zero_r - src_r1
    rel_c = zero_c - src_c1
    
    # Destination section
    dst_r1, dst_r2 = row_ranges[rel_r]
    dst_c1, dst_c2 = col_ranges[rel_c]
    
    # Extract source section, replace separator color with bg
    sec_h = src_r2 - src_r1 + 1
    sec_w = src_c2 - src_c1 + 1
    content = []
    for r in range(sec_h):
        row = []
        for c in range(sec_w):
            v = grid[src_r1 + r][src_c1 + c]
            if v == sep_color and sep_color != bg:
                v = bg
            row.append(v)
        content.append(row)
    
    # Build output
    out = [[bg] * cols for _ in range(rows)]
    
    # Place separators
    for r in sep_rows:
        for c in range(cols):
            out[r][c] = sep_color
    for c in sep_cols:
        for r in range(rows):
            out[r][c] = sep_color
    
    # Place content in destination
    for r in range(sec_h):
        for c in range(sec_w):
            out[dst_r1 + r][dst_c1 + c] = content[r][c]
    
    return out
