def transform(input_grid):
    """
    Rule: The grid has line separators (full rows or full columns of a non-background color)
    dividing it into sections. Isolated single-cell "dots" in the background create an
    expanding diamond toward each adjacent line. The diamond widens by 1 cell per step,
    fills with the line color, marks the dot's position on the line with the dot color,
    and mirrors symmetrically on the other side of the line.
    """
    import copy
    from collections import Counter

    grid = input_grid
    rows = len(grid)
    cols = len(grid[0])

    # Background = most frequent color
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Horizontal line rows: entire row is one non-bg color
    h_lines = set()
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != bg:
            h_lines.add(r)

    # Vertical line columns: entire column is one non-bg color
    v_lines = set()
    for c in range(cols):
        col_vals = set(grid[r][c] for r in range(rows))
        if len(col_vals) == 1 and grid[0][c] != bg:
            v_lines.add(c)

    # Dots: non-bg cells not on any line row or line column
    dots = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and r not in h_lines and c not in v_lines:
                dots.append((r, c, grid[r][c]))

    output = copy.deepcopy(grid)
    sorted_h = sorted(h_lines)
    sorted_v = sorted(v_lines)

    for dr, dc, dot_color in dots:
        # Find nearest h-line above and below
        h_above = None
        for h in sorted_h:
            if h < dr:
                h_above = h
        h_below = None
        for h in sorted_h:
            if h > dr:
                h_below = h
                break

        # Find nearest v-line left and right
        v_left = None
        for v in sorted_v:
            if v < dc:
                v_left = v
        v_right = None
        for v in sorted_v:
            if v > dc:
                v_right = v
                break

        # Expand toward each adjacent horizontal line
        for h_line in [h_above, h_below]:
            if h_line is None:
                continue
            line_color = grid[h_line][0]
            D = abs(dr - h_line)
            direction = 1 if h_line > dr else -1

            # Expanding triangle from dot toward line
            for d in range(1, D):
                r = dr + d * direction
                for c_off in range(-d, d + 1):
                    cc = dc + c_off
                    if 0 <= cc < cols:
                        output[r][cc] = line_color

            # Mark dot's column on the line with dot color
            output[h_line][dc] = dot_color

            # Mirror on other side of line
            for d in range(1, D):
                spread = d
                dist_from_line = D - d
                r_mirror = h_line + direction * dist_from_line
                if 0 <= r_mirror < rows:
                    for c_off in range(-spread, spread + 1):
                        cc = dc + c_off
                        if 0 <= cc < cols:
                            output[r_mirror][cc] = line_color

        # Expand toward each adjacent vertical line
        for v_line in [v_left, v_right]:
            if v_line is None:
                continue
            line_color = grid[0][v_line]
            D = abs(dc - v_line)
            direction = 1 if v_line > dc else -1

            # Expanding triangle from dot toward line
            for d in range(1, D):
                c = dc + d * direction
                for r_off in range(-d, d + 1):
                    rr = dr + r_off
                    if 0 <= rr < rows:
                        output[rr][c] = line_color

            # Mark dot's row on the line with dot color
            output[dr][v_line] = dot_color

            # Mirror on other side of line
            for d in range(1, D):
                spread = d
                dist_from_line = D - d
                c_mirror = v_line + direction * dist_from_line
                if 0 <= c_mirror < cols:
                    for r_off in range(-spread, spread + 1):
                        rr = dr + r_off
                        if 0 <= rr < rows:
                            output[rr][c_mirror] = line_color

    return output
