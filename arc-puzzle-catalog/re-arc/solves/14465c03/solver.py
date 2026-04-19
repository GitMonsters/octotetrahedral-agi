def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0])
    
    sep_rows = [r for r in range(R) if len(set(input_grid[r])) == 1]
    if not sep_rows:
        return input_grid
    sep_val = input_grid[sep_rows[0]][0]
    
    cell_h = sep_rows[0]
    sep_cols = []
    for cell_w in range(1, C):
        candidate_seps, pos, valid = [], cell_w, True
        while pos < C:
            if all(input_grid[r][pos] == sep_val for r in range(R)):
                candidate_seps.append(pos)
                pos += cell_w + 1
            else:
                valid = False
                break
        if valid and candidate_seps and C - (candidate_seps[-1] + 1) == cell_w:
            sep_cols = candidate_seps
            break
    
    from collections import Counter
    vals = [input_grid[r][c] for r in range(R) for c in range(C) 
            if r not in sep_rows and c not in sep_cols]
    bg_val = Counter(vals).most_common(1)[0][0]
    
    row_ranges, prev = [], 0
    for sr in sep_rows:
        if sr > prev: row_ranges.append((prev, sr))
        prev = sr + 1
    if prev < R: row_ranges.append((prev, R))
    
    col_ranges, prev = [], 0
    for sc in sep_cols:
        if sc > prev: col_ranges.append((prev, sc))
        prev = sc + 1
    if prev < C: col_ranges.append((prev, C))
    
    seven_cell = seven_pos = None
    cells_content = {}
    for ri, (r1, r2) in enumerate(row_ranges):
        for ci, (c1, c2) in enumerate(col_ranges):
            cell = [[input_grid[r][c] for c in range(c1, c2)] for r in range(r1, r2)]
            cells_content[(ri, ci)] = cell
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if input_grid[r][c] == 7:
                        seven_cell = (ri, ci)
                        seven_pos = (r - r1, c - c1)
    
    # Determine if sep_val should be cleaned from 7-cell
    # Rule: if any non-7 non-empty cell has ONLY sep_val (no other non-bg values),
    # then sep is an independent noise pattern → DON'T clean
    # Otherwise → DO clean
    should_clean = False
    if sep_val != bg_val:
        has_sep_only_cell = False
        for key, cell in cells_content.items():
            if key == seven_cell:
                continue
            ch, cw = len(cell), len(cell[0])
            non_bg = [(r, c, cell[r][c]) for r in range(ch) for c in range(cw) if cell[r][c] != bg_val]
            if not non_bg:
                continue
            has_non_sep = any(v != sep_val for _, _, v in non_bg)
            if not has_non_sep:
                has_sep_only_cell = True
                break
        should_clean = not has_sep_only_cell
    
    src_cell = cells_content[seven_cell]
    cleaned = []
    for row in src_cell:
        cleaned.append([bg_val if (v == sep_val and should_clean) else v for v in row])
    
    dst_ri, dst_ci = seven_pos
    dst_r1, dst_r2 = row_ranges[dst_ri]
    dst_c1, dst_c2 = col_ranges[dst_ci]
    
    output = [[bg_val] * C for _ in range(R)]
    for r in sep_rows:
        output[r] = [sep_val] * C
    for c in sep_cols:
        for r in range(R):
            output[r][c] = sep_val
    
    for dr in range(len(cleaned)):
        for dc in range(len(cleaned[0])):
            output[dst_r1 + dr][dst_c1 + dc] = cleaned[dr][dc]
    
    return output
