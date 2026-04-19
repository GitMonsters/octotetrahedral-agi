def transform(grid):
    R, C = len(grid), len(grid[0])
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg.setdefault(grid[r][c], []).append((r, c))

    out = [row[:] for row in grid]

    if len(non_bg) == 1:
        color = list(non_bg.keys())[0]
        cells = non_bg[color]
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        filled_rows = set(rs)
        filled_cols = set(cs)
        unfilled_rows = len([r for r in range(min_r, max_r + 1) if r not in filled_rows])
        unfilled_cols = len([c for c in range(min_c, max_c + 1) if c not in filled_cols])

        cr = max_r + 0.5 + unfilled_rows / 2
        cc = max_c + 0.5 + unfilled_cols / 2

        for r, c in cells:
            for rr, cc_v in [(r, c), (r, int(2 * cc - c)), (int(2 * cr - r), c), (int(2 * cr - r), int(2 * cc - c))]:
                if 0 <= rr < R and 0 <= cc_v < C:
                    out[rr][cc_v] = color
    else:
        colors_sorted = sorted(non_bg.keys(), key=lambda k: len(non_bg[k]))
        marker_color = colors_sorted[0]
        shape_color = colors_sorted[1]

        marker_cells = non_bg[marker_color]
        mrs = [r for r, c in marker_cells]
        mcs = [c for r, c in marker_cells]
        cr = (min(mrs) + max(mrs)) / 2
        cc = (min(mcs) + max(mcs)) / 2

        for r, c in non_bg[shape_color]:
            for rr, cc_v in [(r, c), (r, int(2 * cc - c)), (int(2 * cr - r), c), (int(2 * cr - r), int(2 * cc - c))]:
                if 0 <= rr < R and 0 <= cc_v < C:
                    out[rr][cc_v] = shape_color

    return out
