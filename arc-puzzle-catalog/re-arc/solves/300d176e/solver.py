from collections import Counter

def transform(input_grid):
    R = len(input_grid)
    C = len(input_grid[0])
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]

    cells = {}
    for r in range(R):
        for c in range(C):
            if input_grid[r][c] != bg:
                cells[(r, c)] = input_grid[r][c]

    if not cells:
        return input_grid

    non_empty_rows = sorted(set(r for r, c in cells))

    # Group consecutive non-empty rows
    groups = []
    current = []
    for i, r in enumerate(non_empty_rows):
        if current and r > non_empty_rows[i - 1] + 1:
            groups.append(current)
            current = []
        current.append(r)
    if current:
        groups.append(current)

    # Connector color = non-bg color present in the most groups
    colors = set(cells.values())
    if len(colors) <= 1:
        connector = None
    else:
        color_group_count = Counter()
        for g in groups:
            g_colors = set(cells[(r, c)] for r in g for c in range(C) if (r, c) in cells)
            for col in g_colors:
                color_group_count[col] += 1
        connector = color_group_count.most_common(1)[0][0]

    # Classify groups: excluded if connector-only AND single row (isolated waypoint)
    included_groups = []
    excluded_groups = []
    for g in groups:
        g_colors = set(cells[(r, c)] for r in g for c in range(C) if (r, c) in cells)
        if connector is not None:
            non_conn = g_colors - {connector}
            if not non_conn and len(g) == 1:
                excluded_groups.append(g)
            else:
                included_groups.append(g)
        else:
            included_groups.append(g)

    # Recolor connector cells: adjacent to non-connector shape -> take that color
    new_cells = {}
    for (r, c), v in cells.items():
        if connector is not None and v == connector:
            neighbor_color = None
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells and cells[(nr, nc)] != connector:
                    neighbor_color = cells[(nr, nc)]
                    break
            if neighbor_color is not None:
                new_cells[(r, c)] = neighbor_color
            else:
                in_included = any(r in g for g in included_groups)
                if in_included:
                    new_cells[(r, c)] = v
        else:
            new_cells[(r, c)] = v

    def get_connector_cells_in_row(g, row_idx):
        r = g[row_idx]
        return [c for c in range(C) if (r, c) in cells and cells[(r, c)] == connector]

    # Column shifts: bottom included group is anchor (shift=0), propagate upward
    shifts = [0] * len(included_groups)

    if len(included_groups) > 1 and connector is not None:
        for i in range(len(included_groups) - 2, -1, -1):
            g_above = included_groups[i]
            g_below = included_groups[i + 1]

            bot_tips = get_connector_cells_in_row(g_above, -1)
            top_tips = get_connector_cells_in_row(g_below, 0)

            if bot_tips and top_tips:
                ba_col = bot_tips[0]
                tb_col = top_tips[0]
                if len(bot_tips) == 1 and len(top_tips) > 1:
                    tb_col = min(top_tips, key=lambda tc: abs(tc + shifts[i + 1] - ba_col))
                elif len(top_tips) == 1 and len(bot_tips) > 1:
                    ba_col = min(bot_tips, key=lambda bc: abs(bc - (tb_col + shifts[i + 1])))
                shifts[i] = tb_col + shifts[i + 1] - ba_col

    # Overall shift for excluded groups or single-shape cases
    overall_shift = 0
    if excluded_groups:
        last_inc = included_groups[-1]
        first_exc = excluded_groups[0]
        inc_tips = get_connector_cells_in_row(last_inc, -1)
        exc_row = first_exc[0]
        exc_tips = [c for c in range(C) if (exc_row, c) in cells]
        if inc_tips and exc_tips:
            overall_shift = -(exc_tips[0] - inc_tips[0])
    elif len(included_groups) == 1 and connector is None:
        g = included_groups[0]
        all_cols = [c for r in g for c in range(C) if (r, c) in cells]
        min_col = min(all_cols)
        max_col = max(all_cols)
        dist_left = min_col
        dist_right = C - 1 - max_col
        if dist_right <= dist_left:
            overall_shift = dist_right
        else:
            overall_shift = -dist_left

    for i in range(len(shifts)):
        shifts[i] += overall_shift

    # Build row mapping: compact by removing empty rows
    row_mapping = {}
    out_r = 0
    for gi, g in enumerate(included_groups):
        for r in g:
            row_mapping[r] = (out_r, shifts[gi])
            out_r += 1
    content_rows_count = out_r

    # Output dimensions
    all_non_empty = len(non_empty_rows)
    first_non_empty = non_empty_rows[0]

    if len(included_groups) <= 2:
        out_R = all_non_empty + first_non_empty * 2
    else:
        out_R = content_rows_count

    out_C = C
    output = [[bg] * out_C for _ in range(out_R)]

    for (r, c), v in new_cells.items():
        if r in row_mapping:
            out_row, shift = row_mapping[r]
            new_c = c + shift
            if 0 <= out_row < out_R and 0 <= new_c < out_C:
                output[out_row][new_c] = v

    # For single-shape: extend last row if narrower than second-to-last
    if connector is None and len(included_groups) == 1 and content_rows_count >= 2:
        last_r = content_rows_count - 1
        prev_r = content_rows_count - 2
        last_cs = set(c for c in range(out_C) if output[last_r][c] != bg)
        prev_cs = set(c for c in range(out_C) if output[prev_r][c] != bg)
        if last_cs and prev_cs and last_cs < prev_cs:
            for c in prev_cs - last_cs:
                output[last_r][c] = output[prev_r][c]

    return output
