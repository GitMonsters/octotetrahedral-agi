from collections import defaultdict


def transform(grid):
    nrows = len(grid)
    ncols = len(grid[0])
    s = 2 * (nrows - 1)

    complete_diags = []

    for k in range(-(nrows - 1), ncols):
        cells = [(r, r + k) for r in range(nrows) if 0 <= r + k < ncols]
        if len(cells) >= 2:
            vals = [grid[r][c] for r, c in cells]
            if len(set(vals)) == 1:
                complete_diags.append(('nwse', k, vals[0], len(cells)))

    for k in range(0, nrows + ncols - 1):
        cells = [(r, k - r) for r in range(nrows) if 0 <= k - r < ncols]
        if len(cells) >= 2:
            vals = [grid[r][c] for r, c in cells]
            if len(set(vals)) == 1:
                complete_diags.append(('nesw', k, vals[0], len(cells)))

    phase_diags = defaultdict(list)
    for dtype, k, color, length in complete_diags:
        phase_diags[k % s].append((dtype, k, color, length))

    best_phase = None
    for phase, diags in phase_diags.items():
        if len(diags) >= 3 and all(c == 4 for _, _, c, _ in diags):
            if best_phase is None or len(diags) > len(phase_diags.get(best_phase, [])):
                best_phase = phase
    if best_phase is None:
        best_phase = max(phase_diags, key=lambda p: len(phase_diags[p]))

    diags = phase_diags[best_phase]
    nwse_set = sorted(set(k for t, k, c, l in diags if t == 'nwse'))
    nesw_set = sorted(set(k for t, k, c, l in diags if t == 'nesw'))

    nesw_lookup = set(nesw_set)
    v_list = [(d, d + s) for d in nwse_set if d + s in nesw_lookup]

    d_min = min(nwse_set)
    d_max = max(nwse_set)
    e_min = min(nesw_set)
    e_max = max(nesw_set)

    out = [row[:] for row in grid]
    for r in range(nrows):
        for c in range(ncols):
            cr = c - r
            cpr = c + r
            fill = False
            if cr < d_min and cpr < e_min:
                fill = True
            if cr > d_max and cpr > e_max:
                fill = True
            for i, (d, e) in enumerate(v_list):
                if 0 < i < len(v_list) - 1:
                    if cr > d and cpr < e:
                        fill = True
            if fill:
                out[r][c] = 4
    return out
