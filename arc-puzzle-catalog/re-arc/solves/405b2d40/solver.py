def transform(input_grid):
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    counts = Counter()
    for row in input_grid:
        for val in row:
            counts[val] += 1
    bg = counts.most_common(1)[0][0]

    # Find non-background dots
    dots = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                dots.append((r, c, input_grid[r][c]))

    output = copy.deepcopy(input_grid)

    def draw_cross(r, c, color, dr, dc, extend):
        # Stem: extend-1 cells from dot (including dot itself)
        for i in range(extend - 1):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = color
        # Bar at position extend-1 from dot, perpendicular, half-width 2
        bar_r = r + (extend - 1) * dr
        bar_c = c + (extend - 1) * dc
        if dr != 0:  # vertical movement -> horizontal bar
            for j in range(-2, 3):
                nc2 = bar_c + j
                if 0 <= bar_r < rows and 0 <= nc2 < cols:
                    output[bar_r][nc2] = color
        else:  # horizontal movement -> vertical bar
            for j in range(-2, 3):
                nr2 = bar_r + j
                if 0 <= nr2 < rows and 0 <= bar_c < cols:
                    output[nr2][bar_c] = color
        # Wing tips at position extend from dot, at bar extremes (±2)
        wing_r = r + extend * dr
        wing_c = c + extend * dc
        if dr != 0:  # vertical -> wings at horizontal extremes
            for dj in [-2, 2]:
                nc2 = wing_c + dj
                if 0 <= wing_r < rows and 0 <= nc2 < cols:
                    output[wing_r][nc2] = color
        else:  # horizontal -> wings at vertical extremes
            for dj in [-2, 2]:
                nr2 = wing_r + dj
                if 0 <= nr2 < rows and 0 <= wing_c < cols:
                    output[nr2][wing_c] = color

    if len(dots) == 2:
        (r1, c1, col1), (r2, c2, col2) = dots
        if r1 == r2:  # same row
            dist = abs(c2 - c1)
            ext = dist // 2
            draw_cross(r1, c1, col1, 0, 1 if c2 > c1 else -1, ext)
            draw_cross(r2, c2, col2, 0, 1 if c1 > c2 else -1, ext)
        else:  # same column
            dist = abs(r2 - r1)
            ext = dist // 2
            draw_cross(r1, c1, col1, 1 if r2 > r1 else -1, 0, ext)
            draw_cross(r2, c2, col2, 1 if r1 > r2 else -1, 0, ext)
    elif len(dots) == 1:
        r, c, color = dots[0]
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        dr = center_r - r
        dc = center_c - c
        if abs(dr) >= abs(dc):
            ext = int(abs(dr))
            draw_cross(r, c, color, 1 if dr > 0 else -1, 0, ext)
        else:
            ext = int(abs(dc))
            draw_cross(r, c, color, 0, 1 if dc > 0 else -1, ext)

    return output
