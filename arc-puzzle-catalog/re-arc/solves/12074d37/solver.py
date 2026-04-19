def transform(grid):
    H, W = len(grid), len(grid[0])

    # Find colors and background
    from collections import Counter
    flat = [c for row in grid for c in row]
    color_counts = Counter(flat)
    bg_color = color_counts.most_common(1)[0][0]
    vals = sorted(color_counts.keys())

    # Find bordered rectangle (border made of a non-background color)
    border_color = None
    rect_pos = None
    for v in vals:
        if v == bg_color:
            continue
        positions = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == v]
        if len(positions) < 6:
            continue
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        top = all(grid[r_min][c] == v for c in range(c_min, c_max + 1))
        bottom = all(grid[r_max][c] == v for c in range(c_min, c_max + 1))
        left = all(grid[r][c_min] == v for r in range(r_min, r_max + 1))
        right = all(grid[r][c_max] == v for r in range(r_min, r_max + 1))
        if top and bottom and left and right:
            border_color = v
            rect_pos = (r_min, r_max, c_min, c_max)
            break

    if border_color is None:
        return [list(row) for row in grid]

    r_min, r_max, c_min, c_max = rect_pos
    rect_h = r_max - r_min + 1
    rect_w = c_max - c_min + 1
    rect = [row[c_min:c_max + 1] for row in grid[r_min:r_max + 1]]

    pattern_colors = [v for v in vals if v != bg_color and v != border_color]

    if pattern_colors:
        # 3-color case: find transformed pattern matches and stamp copies
        pattern_color = pattern_colors[0]

        def rotate90(m):
            return [list(row) for row in zip(*m[::-1])]

        def flipud(m):
            return [row[:] for row in m[::-1]]

        def fliplr(m):
            return [row[::-1] for row in m]

        def get_transforms(m):
            r1 = rotate90(m)
            r2 = rotate90(r1)
            r3 = rotate90(r2)
            fl = fliplr(m)
            results = [m, r1, r2, r3, fl, flipud(m), rotate90(fl), rotate90(rotate90(rotate90(fl)))]
            return results

        transforms = get_transforms(rect)
        output = [row[:] for row in grid]

        for t_rect in transforms:
            th = len(t_rect)
            tw = len(t_rect[0])
            pat_pos = set()
            non_pat_pos = set()
            for r in range(th):
                for c in range(tw):
                    if t_rect[r][c] == pattern_color:
                        pat_pos.add((r, c))
                    else:
                        non_pat_pos.add((r, c))

            for r0 in range(H - th + 1):
                for c0 in range(W - tw + 1):
                    if r0 == r_min and c0 == c_min and th == rect_h and tw == rect_w:
                        continue
                    ok = True
                    for r, c in pat_pos:
                        if grid[r0 + r][c0 + c] != pattern_color:
                            ok = False
                            break
                    if ok:
                        for r, c in non_pat_pos:
                            if grid[r0 + r][c0 + c] != bg_color:
                                ok = False
                                break
                    if ok:
                        for r in range(th):
                            for c in range(tw):
                                output[r0 + r][c0 + c] = t_rect[r][c]

        return output

    else:
        # 2-color case: fractal propagation from the rectangle
        L = max(rect_h, rect_w)
        S = min(rect_h, rect_w)
        is_vertical = rect_h >= rect_w

        V_A = [('V', -(S - 2), 2 * L), ('H', L + S - 1, -(S - 1))]
        V_B = [('V', -S, S + 1), ('H', L - 1, S)]
        H_A = [('V', 2 * S, L - S + 1), ('V', -(S + 1), S - 2)]
        H_B = [('H', 2 * S, -(S - 1)), ('V', S, 2 * L + 1)]

        def get_dims(orient):
            return (L, S) if orient == 'V' else (S, L)

        def has_overlap(orient, r, c, placed_list):
            h, w = get_dims(orient)
            if r < 0 or c < 0 or r + h > H or c + w > W:
                return True
            for po, pr, pc in placed_list:
                ph, pw = get_dims(po)
                if r < pr + ph and r + h > pr and c < pc + pw and c + w > pc:
                    return True
            return False

        orig_orient = 'V' if is_vertical else 'H'
        placed = [(orig_orient, r_min, c_min)]
        queue = [(orig_orient, 'A', r_min, c_min)]

        while queue:
            orient, set_name, r, c = queue.pop(0)
            offsets = {'V': {'A': V_A, 'B': V_B}, 'H': {'A': H_A, 'B': H_B}}
            children = offsets[orient][set_name]
            next_set = 'B' if set_name == 'A' else 'A'
            for child_orient, dr, dc in children:
                nr, nc = r + dr, c + dc
                if not has_overlap(child_orient, nr, nc, placed):
                    placed.append((child_orient, nr, nc))
                    queue.append((child_orient, next_set, nr, nc))

        output = [row[:] for row in grid]
        for orient, r, c in placed:
            h, w = get_dims(orient)
            for ri in range(h):
                for ci in range(w):
                    if ri == 0 or ri == h - 1 or ci == 0 or ci == w - 1:
                        output[r + ri][c + ci] = border_color
                    else:
                        output[r + ri][c + ci] = bg_color

        return output
