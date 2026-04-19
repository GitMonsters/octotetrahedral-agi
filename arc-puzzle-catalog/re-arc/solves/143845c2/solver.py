"""Solver for ARC-AGI task 143845c2.

The transformation rule:
- Output is 3x the input in each dimension
- Background (most common value) stays as bg in the output
- Foreground cells are redistributed into a smooth blob shape
- The shape is determined by the cumulative distribution of fg cells
"""

def transform(input_grid):
    """Transform input grid according to task 143845c2 rules."""
    H = len(input_grid)
    W = len(input_grid[0])
    oH, oW = 3 * H, 3 * W

    # Determine bg (most common value)
    from collections import Counter
    counts = Counter()
    for row in input_grid:
        for v in row:
            counts[v] += 1
    bg = counts.most_common(1)[0][0]

    # Determine fg value (most common non-bg, ignoring 9)
    fg_counts = Counter()
    for row in input_grid:
        for v in row:
            if v != bg and v != 9:
                fg_counts[v] += 1
    if fg_counts:
        fg_val = fg_counts.most_common(1)[0][0]
    else:
        fg_val = 9  # fallback

    # Build binary mask (all non-bg cells are fg)
    mask = [[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg:
                mask[r][c] = 1

    N = sum(mask[r][c] for r in range(H) for c in range(W))
    if N == 0:
        return [[bg]*oW for _ in range(oH)]

    # Cumulative row sums
    row_sums = [sum(mask[r]) for r in range(H)]
    cum_row = [0]
    for rs in row_sums:
        cum_row.append(cum_row[-1] + rs)

    # Cumulative col sums
    col_sums = [sum(mask[r][c] for r in range(H)) for c in range(W)]
    cum_col = [0]
    for cs in col_sums:
        cum_col.append(cum_col[-1] + cs)

    # 2D prefix sum
    P = [[0]*(W+1) for _ in range(H+1)]
    for r in range(H):
        for c in range(W):
            P[r+1][c+1] = mask[r][c] + P[r][c+1] + P[r+1][c] - P[r][c]

    def interp_2d(r, c):
        r = max(0.0, min(float(H), r))
        c = max(0.0, min(float(W), c))
        r0 = int(r); c0 = int(c)
        r1 = min(r0+1, H); c1 = min(c0+1, W)
        fr = r - r0; fc = c - c0
        return (P[r0][c0]*(1-fr)*(1-fc) + P[r1][c0]*fr*(1-fc) +
                P[r0][c1]*(1-fr)*fc + P[r1][c1]*fr*fc)

    def cr_interp(x):
        x = max(0.0, min(float(H), x))
        i = int(x)
        if i >= H:
            return float(cum_row[H])
        return cum_row[i] + (x - i) * (cum_row[i+1] - cum_row[i])

    def cc_interp(x):
        x = max(0.0, min(float(W), x))
        i = int(x)
        if i >= W:
            return float(cum_col[W])
        return cum_col[i] + (x - i) * (cum_col[i+1] - cum_col[i])

    def compute_features(R, C):
        r = R / 3.0
        c = C / 3.0

        cr = cr_interp(r) / N
        cc = cc_interp(c) / N
        p_rc = interp_2d(r, c) / N

        nw = p_rc
        ne = cr - p_rc
        sw = cc - p_rc
        se = 1 - cr - cc + p_rc

        cr_h = cr_interp(r + 0.5) / N
        cc_h = cc_interp(c + 0.5) / N
        p_h = interp_2d(r + 0.5, c + 0.5) / N

        nw_h = p_h
        ne_h = cr_h - p_h
        sw_h = cc_h - p_h
        se_h = 1 - cr_h - cc_h + p_h

        nw_se = nw * se
        ne_sw = ne * sw
        nwh_seh = nw_h * se_h
        neh_swh = ne_h * sw_h
        crh_1mcch = cr_h * (1 - cc_h)
        cch_1mcrh = cc_h * (1 - cr_h)

        denom = max(0.001, (cr_h * (1-cr_h) * cc_h * (1-cc_h))**0.5)
        corr_h = (p_h - cr_h * cc_h) / denom

        return [
            nw, ne, sw, se, cr, cc, p_rc,               # 0-6
            nw_se, ne_sw, nw_se - ne_sw,                 # 7-9
            cr_h, cc_h, p_h,                              # 10-12
            nw_h, ne_h, sw_h, se_h,                       # 13-16
            nwh_seh, neh_swh, nwh_seh - neh_swh,         # 17-19
            cr + cc, abs(cr - cc),                         # 20-21
            cr_h + cc_h, abs(cr_h - cc_h),                 # 22-23
            min(cr, cc), max(cr, cc),                      # 24-25
            min(1-cr, 1-cc), max(1-cr, 1-cc),              # 26-27
            min(cr_h, cc_h), max(cr_h, cc_h),              # 28-29
            min(1-cr_h, 1-cc_h), max(1-cr_h, 1-cc_h),     # 30-31
            cr * (1-cc), cc * (1-cr),                      # 32-33
            crh_1mcch, cch_1mcrh,                          # 34-35
            crh_1mcch - cch_1mcrh,                         # 36
            cr*(1-cc) - cc*(1-cr),                         # 37
            cr - R/(3.0*H), cc - C/(3.0*W),               # 38-39
            nwh_seh + neh_swh,                             # 40
            (nw_h + se_h) - (ne_h + sw_h),                # 41
            p_h - cr_h * cc_h,                             # 42
            corr_h,                                        # 43
            min(nwh_seh, neh_swh),                         # 44
            max(nwh_seh, neh_swh),                         # 45
        ]

    def classify(f):
        if f[26] <= 0.3222222328:
            if f[39] <= -0.0277777780:
                if f[20] <= 1.2500000000:
                    if f[38] <= 0.1666666679:
                        if f[5] <= 0.4999999925:
                            return 0
                        else:
                            return 1
                    else:
                        if f[6] <= 0.2222222313:
                            return 1
                        else:
                            return 0
                else:
                    if f[36] <= -0.1666666679:
                        return 1
                    else:
                        return 0
            else:
                if f[44] <= 0.0136419754:
                    if f[7] <= 0.0268587107:
                        if f[7] <= 0.0195336081:
                            return 0
                        else:
                            if f[7] <= 0.0200274354:
                                if f[37] <= -0.3333333358:
                                    return 1
                                else:
                                    return 0
                            else:
                                return 0
                    else:
                        if f[17] <= 0.0099657066:
                            if f[28] <= 0.5333333313:
                                return 1
                            else:
                                return 0
                        else:
                            return 0
                else:
                    if f[2] <= 0.3888888955:
                        return 0
                    else:
                        return 1
        else:
            if f[36] <= 0.1388888955:
                if f[18] <= 0.0106138550:
                    if f[43] <= 0.2503455207:
                        return 0
                    else:
                        return 1
                else:
                    if f[4] <= 0.5888889134:
                        if f[39] <= -0.2500000075:
                            return 0
                        else:
                            if f[20] <= 1.2111111283:
                                if f[30] <= 0.8444444239:
                                    if f[34] <= 0.0483796299:
                                        if f[43] <= 0.0044944840:
                                            return 1
                                        else:
                                            return 0
                                    else:
                                        return 1
                                else:
                                    return 0
                            else:
                                return 0
                    else:
                        return 0
            else:
                if f[38] <= 0.0370370373:
                    if f[30] <= 0.7194444537:
                        return 1
                    else:
                        return 0
                else:
                    if f[16] <= 0.1131944433:
                        if f[0] <= 0.1833333373:
                            if f[39] <= -0.0555555560:
                                return 1
                            else:
                                return 0
                        else:
                            return 0
                    else:
                        return 0

    # Build output
    output = [[bg]*oW for _ in range(oH)]
    for R in range(oH):
        for C in range(oW):
            f = compute_features(R, C)
            if classify(f) == 1:
                output[R][C] = fg_val

    return output
