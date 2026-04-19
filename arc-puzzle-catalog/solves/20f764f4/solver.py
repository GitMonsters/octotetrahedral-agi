from collections import Counter

def find_content_rows_rule(grid, bg, start_row, direction, content_cols):
    """Find content rows using 2nd-all-bg rule"""
    H = len(grid)
    rows = []
    if direction == 'down':
        scan = range(start_row, H)
    else:
        scan = range(start_row, -1, -1)
    
    found_non_bg = False
    all_bg_count = 0
    result = []
    for r in scan:
        is_all_bg = all(grid[r][c] == bg for c in content_cols)
        if direction == 'up':
            result.insert(0, r)
        else:
            result.append(r)
        if not is_all_bg:
            found_non_bg = True
        elif found_non_bg:
            all_bg_count += 1
            if all_bg_count >= 2:
                break
    return result

def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    
    # Find sep rows/cols iteratively with 90% threshold
    sep_rows = set()
    sep_cols = set()
    changed = True
    while changed:
        changed = False
        for r in range(H):
            if r in sep_rows: continue
            vals = [grid[r][c] for c in range(W) if c not in sep_cols]
            if not vals: continue
            cnt = Counter(vals)
            tv, tc = cnt.most_common(1)[0]
            if tv != bg and tc / len(vals) >= 0.9:
                sep_rows.add(r); changed = True
        for c in range(W):
            if c in sep_cols: continue
            vals = [grid[r][c] for r in range(H) if r not in sep_rows]
            if not vals: continue
            cnt = Counter(vals)
            tv, tc = cnt.most_common(1)[0]
            if tv != bg and tc / len(vals) >= 0.9:
                sep_cols.add(c); changed = True
    
    sep_rows = sorted(sep_rows)
    sep_cols = sorted(sep_cols)
    
    def get_sep_val(r_or_c, axis):
        if axis == 'row':
            vals = [grid[r_or_c][c] for c in range(W) if c not in set(sep_cols)]
        else:
            vals = [grid[r][r_or_c] for r in range(H) if r not in set(sep_rows)]
        return Counter(vals).most_common(1)[0][0] if vals else None
    
    sep_row_vals = {r: get_sep_val(r, 'row') for r in sep_rows}
    sep_col_vals = {c: get_sep_val(c, 'col') for c in sep_cols}
    
    # Find content_val: sep value that also appears in content cells
    all_sep_vals = set(sep_row_vals.values()) | set(sep_col_vals.values())
    candidate_counts = Counter()
    for r in range(H):
        if r in set(sep_rows): continue
        for c in range(W):
            if c in set(sep_cols): continue
            v = grid[r][c]
            if v != bg and v in all_sep_vals:
                candidate_counts[v] += 1
    
    if not candidate_counts:
        return [row[:] for row in grid]
    
    content_val = candidate_counts.most_common(1)[0][0]
    
    # Target seps (prefer row sep over col sep)
    target_row_seps = [r for r, v in sep_row_vals.items() if v == content_val]
    target_col_seps = [c for c, v in sep_col_vals.items() if v == content_val]
    
    # Content cols = between the two bounding col seps (middle group)
    if len(sep_cols) >= 2:
        left_sep_col = sep_cols[0]
        right_sep_col = sep_cols[-1]
        content_cols = list(range(left_sep_col + 1, right_sep_col))
    elif len(sep_cols) == 1:
        sc = sep_cols[0]
        if sc < W // 2:
            left_sep_col = sc
            right_sep_col = None
            content_cols = list(range(sc + 1, W))
        else:
            left_sep_col = None
            right_sep_col = sc
            content_cols = list(range(0, sc))
    else:
        return [row[:] for row in grid]
    
    # Determine fill direction and content rows
    fill_dir = None
    content_rows = []
    top_sep_row = None
    bot_sep_row = None
    
    if target_row_seps:
        target_r = target_row_seps[0]
        other_row_seps = sorted([r for r in sep_rows if r != target_r])
        
        if len(sep_rows) >= 2:
            # Find row group between two seps that contains content_val
            srl = sorted(sep_rows)
            for i in range(len(srl) - 1):
                r1, r2 = srl[i], srl[i+1]
                crows = list(range(r1 + 1, r2))
                if any(grid[r][c] == content_val for r in crows for c in content_cols):
                    top_sep_row = r1
                    bot_sep_row = r2
                    content_rows = crows
                    fill_dir = 'vert_down' if target_r == r2 else 'vert_up'
                    break
        else:
            # One row sep
            if target_r < H // 2:
                # Content below, fill upward toward target
                fill_dir = 'vert_up'
                top_sep_row = target_r
                content_rows = find_content_rows_rule(grid, bg, target_r + 1, 'down', content_cols)
                bot_sep_row = None
            else:
                fill_dir = 'vert_down'
                bot_sep_row = target_r
                content_rows = find_content_rows_rule(grid, bg, target_r - 1, 'up', content_cols)
                top_sep_row = None
    
    elif target_col_seps:
        target_c = target_col_seps[0]
        fill_dir = 'horiz_right' if target_c == right_sep_col else 'horiz_left'
        
        if len(sep_rows) >= 2:
            srl = sorted(sep_rows)
            for i in range(len(srl) - 1):
                r1, r2 = srl[i], srl[i+1]
                crows = list(range(r1 + 1, r2))
                if any(grid[r][c] == content_val for r in crows for c in content_cols):
                    top_sep_row = r1
                    bot_sep_row = r2
                    content_rows = crows
                    break
        elif len(sep_rows) == 1:
            sr = sep_rows[0]
            top_sep_row = sr
            bot_sep_row = None
            content_rows = find_content_rows_rule(grid, bg, sr + 1, 'down', content_cols)
        else:
            content_rows = list(range(H))
    else:
        return [row[:] for row in grid]
    
    if not content_rows or fill_dir is None:
        return [row[:] for row in grid]
    
    # Apply fill rule
    content_data = [[grid[r][c] for c in content_cols] for r in content_rows]
    filled = [list(row) for row in content_data]
    
    if fill_dir == 'horiz_right':
        for ri, row in enumerate(filled):
            first = next((i for i, v in enumerate(row) if v != bg), None)
            if first is not None:
                fv = row[first]
                for i in range(first, len(row)): filled[ri][i] = fv
    elif fill_dir == 'horiz_left':
        for ri, row in enumerate(filled):
            last = next((i for i, v in enumerate(reversed(row)) if v != bg), None)
            if last is not None:
                last_idx = len(row) - 1 - last
                fv = row[last_idx]
                for i in range(0, last_idx + 1): filled[ri][i] = fv
    elif fill_dir == 'vert_down':
        acc = [bg] * len(content_cols)
        for ri in range(len(filled)):
            for ci in range(len(content_cols)):
                if filled[ri][ci] != bg: acc[ci] = filled[ri][ci]
                else: filled[ri][ci] = acc[ci]
    elif fill_dir == 'vert_up':
        acc = [bg] * len(content_cols)
        for ri in range(len(filled) - 1, -1, -1):
            for ci in range(len(content_cols)):
                if filled[ri][ci] != bg: acc[ci] = filled[ri][ci]
                else: filled[ri][ci] = acc[ci]
    
    # Assemble output
    output = []
    
    if top_sep_row is not None:
        row = []
        if left_sep_col is not None: row.append(grid[top_sep_row][left_sep_col])
        for c in content_cols: row.append(grid[top_sep_row][c])
        if right_sep_col is not None: row.append(grid[top_sep_row][right_sep_col])
        output.append(row)
    
    for ri, r in enumerate(content_rows):
        row = []
        if left_sep_col is not None: row.append(grid[r][left_sep_col])
        row.extend(filled[ri])
        if right_sep_col is not None: row.append(grid[r][right_sep_col])
        output.append(row)
    
    if bot_sep_row is not None:
        row = []
        if left_sep_col is not None: row.append(grid[bot_sep_row][left_sep_col])
        for c in content_cols: row.append(grid[bot_sep_row][c])
        if right_sep_col is not None: row.append(grid[bot_sep_row][right_sep_col])
        output.append(row)
    
    return output
