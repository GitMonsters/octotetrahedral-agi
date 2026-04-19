from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find dividers
    h_divs = []
    for r in range(H):
        counts = Counter(grid[r])
        for val, cnt in counts.most_common():
            if val != bg and cnt >= W * 0.7:
                h_divs.append((r, val))
                break

    v_divs = []
    for c in range(W):
        col = [grid[r][c] for r in range(H)]
        counts = Counter(col)
        for val, cnt in counts.most_common():
            if val != bg and cnt >= H * 0.7:
                v_divs.append((c, val))
                break

    h_rows = {r for r, v in h_divs}
    v_cols = {c for c, v in v_divs}
    h_sorted = sorted(h_divs, key=lambda x: x[0])
    v_sorted = sorted(v_divs, key=lambda x: x[0])

    # Row groups and col groups
    rgs = []
    prev = 0
    for r in sorted(h_rows):
        if r > prev:
            rgs.append((prev, r - 1))
        prev = r + 1
    if prev < H:
        rgs.append((prev, H - 1))

    cgs = []
    prev = 0
    for c in sorted(v_cols):
        if c > prev:
            cgs.append((prev, c - 1))
        prev = c + 1
    if prev < W:
        cgs.append((prev, W - 1))

    # Middle row group (between h_dividers)
    first_h, last_h = h_sorted[0][0], h_sorted[-1][0]
    mid_rg = None
    for rs, re in rgs:
        if rs > first_h and re < last_h:
            mid_rg = (rs, re)
            break
    if mid_rg is None:
        mid_rg = max(rgs, key=lambda x: x[1] - x[0] + 1)

    # Middle col group (between v_dividers, or largest)
    mid_cg = None
    if len(v_sorted) >= 2:
        for cs, ce in cgs:
            if cs > v_sorted[0][0] and ce < v_sorted[-1][0]:
                mid_cg = (cs, ce)
                break
    if mid_cg is None:
        mid_cg = max(cgs, key=lambda x: x[1] - x[0] + 1)

    mid_h = mid_rg[1] - mid_rg[0] + 1
    mid_cw = mid_cg[1] - mid_cg[0] + 1

    # Find mark_val: look for non-bg, non-divider values in non-divider cells
    mark_val = None
    div_vals = set(v for _, v in h_divs) | set(v for _, v in v_divs)
    for r in range(H):
        if r in h_rows:
            continue
        for c in range(W):
            if c in v_cols:
                continue
            if grid[r][c] != bg and grid[r][c] not in div_vals:
                mark_val = grid[r][c]
                break
        if mark_val is not None:
            break

    # If no scattered marks, check if mark_val matches a divider value
    if mark_val is None:
        for r in range(H):
            if r in h_rows:
                continue
            for c in range(W):
                if c in v_cols:
                    continue
                if grid[r][c] != bg:
                    mark_val = grid[r][c]
                    break
            if mark_val is not None:
                break

    # If still None, use h_div with bg holes as mark_val
    if mark_val is None:
        for r, v in h_sorted:
            has_bg = any(grid[r][c] == bg for c in range(W) if c not in v_cols)
            if has_bg:
                mark_val = v
                break
        if mark_val is None and len(h_sorted) >= 2:
            # Use the non-dominant h_div value
            for r, v in h_sorted:
                if not all(grid[r][c] == v for c in range(W)):
                    mark_val = v
                    break
            if mark_val is None:
                mark_val = h_sorted[-1][1]

    # Determine gravity: mark_val matches which divider?
    gravity_down = any(v == mark_val for _, v in h_divs)
    gravity_right = any(v == mark_val for _, v in v_divs)

    # Which v_divs are left/right of mid_cg?
    left_vdivs = [(c, v) for c, v in v_sorted if c < mid_cg[0]]
    right_vdivs = [(c, v) for c, v in v_sorted if c > mid_cg[1]]

    n_v = len(v_divs)
    out_w = 9
    int_w = out_w - n_v
    int_h = mid_h
    out_h = int_h + len(h_divs)

    # Determine which mid_cg positions to show
    if mid_cw > int_w:
        if right_vdivs and not left_vdivs:
            # v_div on right only: drop from LEFT
            col_offset = mid_cw - int_w
        elif left_vdivs and not right_vdivs:
            # v_div on left only: drop from RIGHT
            col_offset = 0
        else:
            col_offset = 0
    else:
        col_offset = 0

    # Map output interior col j -> mid_cg col (mid_cg[0] + col_offset + j)
    # Map output interior row i -> mid_rg row (mid_rg[0] + i)

    # Determine border column positions in output
    border_cols = {}
    out_col_idx = 0
    if left_vdivs:
        border_cols[0] = left_vdivs[0][1]
    if right_vdivs:
        border_cols[out_w - 1] = right_vdivs[-1][1]
    int_cols = [c for c in range(out_w) if c not in border_cols]

    # Determine dominant h_divs
    dominant_h = set()
    for r, v in h_sorted:
        if all(grid[r][c] == v for c in range(W)):
            dominant_h.add(r)

    # Build output
    out = [[bg] * out_w for _ in range(out_h)]

    # Fill interior with shadow pattern
    if mark_val is not None:
        if gravity_down:
            # For each column of mid_cg (mapped), find topmost mark -> bar extends down
            for j in range(min(int_w, mid_cw - col_offset)):
                mc = mid_cg[0] + col_offset + j
                out_col = int_cols[j]
                topmost = -1
                for i in range(mid_h):
                    r = mid_rg[0] + i
                    if grid[r][mc] == mark_val:
                        topmost = i
                        break
                if topmost >= 0:
                    bar_height = mid_h - topmost
                    for k in range(int_h):
                        if k >= int_h - bar_height:
                            out[k + 1][out_col] = mark_val

        elif gravity_right:
            # For each row of mid_rg, find leftmost mark -> bar extends right
            for i in range(int_h):
                r = mid_rg[0] + i
                out_r = i + 1
                leftmost = -1
                for j in range(min(int_w, mid_cw - col_offset)):
                    mc = mid_cg[0] + col_offset + j
                    if grid[r][mc] == mark_val:
                        leftmost = j
                        break
                if leftmost >= 0:
                    bar_width = min(int_w, mid_cw - col_offset) - leftmost
                    for k in range(len(int_cols)):
                        if k >= len(int_cols) - bar_width:
                            out[out_r][int_cols[k]] = mark_val

    # Draw borders
    # V_div border columns
    for col_pos, val in border_cols.items():
        for r in range(out_h):
            out[r][col_pos] = val

    # H_div border rows
    for idx, (inp_r, val) in enumerate(h_sorted):
        out_r = idx  # first h_div -> row 0, second -> last row
        if idx > 0:
            out_r = out_h - 1 - (len(h_sorted) - 1 - idx)

        if inp_r in dominant_h:
            # Dominant: fill entire row
            for c in range(out_w):
                out[out_r][c] = val
        elif val == mark_val:
            # Mark-val h_div: use mapped input row
            for j in range(min(int_w, mid_cw - col_offset)):
                mc = mid_cg[0] + col_offset + j
                out[out_r][int_cols[j]] = grid[inp_r][mc]
            # V_div borders stay as-is (already drawn)
        else:
            # Non-dominant, non-mark: fill with h_div val, keep v_div borders
            for c in range(out_w):
                if c not in border_cols:
                    out[out_r][c] = val

    # Re-apply dominant h_div rows (they override everything)
    for idx, (inp_r, val) in enumerate(h_sorted):
        out_r = idx if idx == 0 else out_h - 1 - (len(h_sorted) - 1 - idx)
        if inp_r in dominant_h:
            for c in range(out_w):
                out[out_r][c] = val

    return out
