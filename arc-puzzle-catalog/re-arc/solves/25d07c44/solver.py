def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Pattern: There's a "spine" (row or column with exactly 2 distinct colors and the most
    non-background cells). Perpendicular to the spine, there are 2 "template" positions
    (one for each spine color) that contain non-bg data. Each perpendicular line's output
    is determined by its values at the 2 template positions, mapped via the spine pattern.
    """
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find spine: row or column with exactly 2 distinct colors and most non-bg cells
    best = None  # (is_row, index, non_bg_count)

    for r in range(rows):
        unique = set(input_grid[r])
        non_bg = sum(1 for v in input_grid[r] if v != bg)
        if len(unique) == 2 and non_bg > 0:
            if best is None or non_bg > best[2]:
                best = (True, r, non_bg)

    for c in range(cols):
        col_vals = [input_grid[r][c] for r in range(rows)]
        unique = set(col_vals)
        non_bg = sum(1 for v in col_vals if v != bg)
        if len(unique) == 2 and non_bg > 0:
            if best is None or non_bg > best[2]:
                best = (False, c, non_bg)

    spine_is_row, spine_idx, _ = best

    if spine_is_row:
        spine = input_grid[spine_idx][:]
        colors = list(set(spine))
        color_A, color_B = colors[0], colors[1]

        # Find one pair column per spine color that has non-bg data outside the spine row
        cA, cB = None, None
        for c in range(cols):
            has_data = any(input_grid[r][c] != bg for r in range(rows) if r != spine_idx)
            if has_data:
                if spine[c] == color_A and cA is None:
                    cA = c
                elif spine[c] == color_B and cB is None:
                    cB = c
            if cA is not None and cB is not None:
                break

        # Build output: for each row, map spine colors using pair values
        output = []
        for r in range(rows):
            a = input_grid[r][cA]
            b = input_grid[r][cB]
            out_row = []
            for c in range(cols):
                if spine[c] == color_A:
                    out_row.append(a)
                else:
                    out_row.append(b)
            output.append(out_row)
        return output

    else:
        spine = [input_grid[r][spine_idx] for r in range(rows)]
        colors = list(set(spine))
        color_A, color_B = colors[0], colors[1]

        # Find one pair row per spine color that has non-bg data outside the spine column
        rA, rB = None, None
        for r in range(rows):
            has_data = any(input_grid[r][c] != bg for c in range(cols) if c != spine_idx)
            if has_data:
                if spine[r] == color_A and rA is None:
                    rA = r
                elif spine[r] == color_B and rB is None:
                    rB = r
            if rA is not None and rB is not None:
                break

        # Build output: for each column, map spine colors using pair values
        output = []
        for r in range(rows):
            out_row = []
            for c in range(cols):
                if spine[r] == color_A:
                    out_row.append(input_grid[rA][c])
                else:
                    out_row.append(input_grid[rB][c])
            output.append(out_row)
        return output
