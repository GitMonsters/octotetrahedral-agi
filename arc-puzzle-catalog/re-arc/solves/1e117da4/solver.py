def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    from collections import Counter, defaultdict

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    by_color = defaultdict(set)
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                by_color[input_grid[r][c]].add((r, c))

    # Find frame: rectangle with 4 same-color corners, bg interior, 2+ colored sides
    frame = None
    for color, cells in by_color.items():
        cell_rows = sorted(set(r for r, c in cells))
        found = False
        for i in range(len(cell_rows)):
            for i2 in range(i + 1, len(cell_rows)):
                r1, r2 = cell_rows[i], cell_rows[i2]
                if r2 - r1 < 2:
                    continue
                cols_r1 = set(c for rr, c in cells if rr == r1)
                cols_r2 = set(c for rr, c in cells if rr == r2)
                common_cols = sorted(cols_r1 & cols_r2)
                for j in range(len(common_cols)):
                    for j2 in range(j + 1, len(common_cols)):
                        c1, c2 = common_cols[j], common_cols[j2]
                        if c2 - c1 < 2:
                            continue
                        ok = all(
                            input_grid[r][c] == bg
                            for r in range(r1 + 1, r2)
                            for c in range(c1 + 1, c2)
                        )
                        if not ok:
                            continue
                        sides = [
                            [input_grid[r][c1] for r in range(r1 + 1, r2)],
                            [input_grid[r][c2] for r in range(r1 + 1, r2)],
                            [input_grid[r1][c] for c in range(c1 + 1, c2)],
                            [input_grid[r2][c] for c in range(c1 + 1, c2)],
                        ]
                        sc = sum(any(v != bg for v in s) for s in sides)
                        if sc >= 1:
                            frame = (r1, r2, c1, c2, color)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break

    r1, r2, c1, c2, corner_color = frame
    fr, fc = r2 - r1 + 1, c2 - c1 + 1
    ih, iw = r2 - r1 - 1, c2 - c1 - 1

    left_s = [input_grid[r][c1] for r in range(r1 + 1, r2)]
    right_s = [input_grid[r][c2] for r in range(r1 + 1, r2)]
    top_s = [input_grid[r1][c] for c in range(c1 + 1, c2)]
    bot_s = [input_grid[r2][c] for c in range(c1 + 1, c2)]

    lc = left_s[0] if left_s and left_s[0] != bg else None
    rc = right_s[0] if right_s and right_s[0] != bg else None
    tc = top_s[0] if top_s and top_s[0] != bg else None
    bc = bot_s[0] if bot_s and bot_s[0] != bg else None

    fb = set()
    for c in range(c1, c2 + 1):
        fb.add((r1, c))
        fb.add((r2, c))
    for r in range(r1, r2 + 1):
        fb.add((r, c1))
        fb.add((r, c2))

    pc = [
        (r, c, input_grid[r][c])
        for r in range(rows)
        for c in range(cols)
        if input_grid[r][c] != bg and (r, c) not in fb
    ]

    pr1 = min(r for r, c, v in pc)
    pr2 = max(r for r, c, v in pc)
    pcc1 = min(c for r, c, v in pc)
    pcc2 = max(c for r, c, v in pc)
    ph, pw = pr2 - pr1 + 1, pcc2 - pcc1 + 1

    pat = [[bg] * pw for _ in range(ph)]
    for r, c, v in pc:
        pat[r - pr1][c - pcc1] = v

    pcols = set(v for _, _, v in pc)

    hflip = False
    if lc is not None and lc in pcols:
        avg = sum(c - pcc1 for r, c, v in pc if v == lc) / sum(1 for _, _, v in pc if v == lc)
        if avg > (pw - 1) / 2:
            hflip = True
    elif rc is not None and rc in pcols:
        avg = sum(c - pcc1 for r, c, v in pc if v == rc) / sum(1 for _, _, v in pc if v == rc)
        if avg < (pw - 1) / 2:
            hflip = True

    vflip = False
    if tc is not None and tc in pcols:
        avg = sum(r - pr1 for r, c, v in pc if v == tc) / sum(1 for _, _, v in pc if v == tc)
        if avg > (ph - 1) / 2:
            vflip = True
    elif bc is not None and bc in pcols:
        avg = sum(r - pr1 for r, c, v in pc if v == bc) / sum(1 for _, _, v in pc if v == bc)
        if avg < (ph - 1) / 2:
            vflip = True

    if vflip:
        pat = pat[::-1]
    if hflip:
        pat = [row[::-1] for row in pat]

    # Pad pattern if smaller than interior
    if ph < ih:
        pad_rows = ih - ph
        if bc is not None and tc is None:
            pat = [[bg] * pw for _ in range(pad_rows)] + pat
        elif tc is not None and bc is None:
            pat = pat + [[bg] * pw for _ in range(pad_rows)]
        else:
            pat = [[bg] * pw for _ in range(pad_rows)] + pat
    if pw < iw:
        pad_cols = iw - pw
        if rc is not None and lc is None:
            pat = [[bg] * pad_cols + row for row in pat]
        elif lc is not None and rc is None:
            pat = [row + [bg] * pad_cols for row in pat]
        else:
            pat = [[bg] * pad_cols + row for row in pat]

    out = [[bg] * fc for _ in range(fr)]
    for dc in range(fc):
        out[0][dc] = input_grid[r1][c1 + dc]
        out[fr - 1][dc] = input_grid[r2][c1 + dc]
    for dr in range(fr):
        out[dr][0] = input_grid[r1 + dr][c1]
        out[dr][fc - 1] = input_grid[r1 + dr][c2]
    for dr in range(ih):
        for dc in range(iw):
            out[dr + 1][dc + 1] = pat[dr][dc]

    return out
