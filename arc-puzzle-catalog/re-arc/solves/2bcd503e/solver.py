from collections import Counter


def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    def bars_left():
        out = []
        for r in range(rows):
            if grid[r][0] != bg:
                c, L = grid[r][0], 0
                while L < cols and grid[r][L] == c:
                    L += 1
                out.append((r, c, L))
        return out

    def bars_right():
        out = []
        for r in range(rows):
            if grid[r][cols - 1] != bg:
                c, L = grid[r][cols - 1], 0
                while L < cols and grid[r][cols - 1 - L] == c:
                    L += 1
                out.append((r, c, L))
        return out

    def bars_top():
        out = []
        for c in range(cols):
            if grid[0][c] != bg:
                col_c, L = grid[0][c], 0
                while L < rows and grid[L][c] == col_c:
                    L += 1
                out.append((c, col_c, L))
        return out

    def bars_bottom():
        out = []
        for c in range(cols):
            if grid[rows - 1][c] != bg:
                col_c, L = grid[rows - 1][c], 0
                while L < rows and grid[rows - 1 - L][c] == col_c:
                    L += 1
                out.append((c, col_c, L))
        return out

    hl, hr = bars_left(), bars_right()
    vt, vb = bars_top(), bars_bottom()

    # Primary direction = whichever edge has more bars
    h_bars = hl if len(hl) >= len(hr) else hr
    h_dir  = 'left' if len(hl) >= len(hr) else 'right'
    v_bars = vb if len(vb) >= len(vt) else vt

    h_lines = {r: color for r, color, L in h_bars}  # row -> color
    v_lines = {c: color for c, color, L in v_bars}  # col -> color
    h_rows = set(h_lines)
    v_cols = set(v_lines)

    # Extra V-col rule:
    # The topmost H-bar's length encodes a V-col (= length for left bars,
    # cols-length for right bars).  When that encoded col is absent from the
    # existing V-lines the bar's inner edge becomes a new bg-coloured V-line.
    topmost_h = min(h_bars, key=lambda x: x[0])
    _, _, Lh = topmost_h
    enc_col  = Lh if h_dir == 'left' else (cols - Lh)
    extra_vc = Lh - 1 if h_dir == 'left' else (cols - Lh)
    if enc_col not in v_cols:
        v_lines[extra_vc] = bg

    # Extra H-row rule:
    # The rightmost V-bar's length encodes an H-row (= length value).
    # When that H-row exists the complementary row (rows - length) becomes
    # a new bg-coloured H-line.
    rightmost_v = max(v_bars, key=lambda x: x[0])
    _, _, Lv = rightmost_v
    if Lv in h_rows:
        h_lines[rows - Lv] = bg

    # Build output: lay down H-lines, then V-lines; intersections become 6
    out = [[bg] * cols for _ in range(rows)]
    for r, color in h_lines.items():
        for c in range(cols):
            out[r][c] = color
    for c, color in v_lines.items():
        for r in range(rows):
            out[r][c] = 6 if r in h_lines else color
    return out
