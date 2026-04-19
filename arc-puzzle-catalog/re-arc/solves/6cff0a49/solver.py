def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])

    bar_color = None
    bar_orient = None
    bar_start = None
    bar_end = None

    # Detect bar via "pure strip": contiguous columns/rows where every cell
    # is the same color, not touching grid edges.  Handles cases where the
    # bar color also appears as background outside the bar.

    # --- vertical bar (pure columns) ---
    col_color = []  # color if column is pure, else None
    for c in range(cols):
        vals = set(input_grid[r][c] for r in range(rows))
        col_color.append(vals.pop() if len(vals) == 1 else None)

    best_vbar = None
    i = 0
    while i < cols:
        if col_color[i] is not None:
            color = col_color[i]
            j = i
            while j < cols and col_color[j] == color:
                j += 1
            if i > 0 and j - 1 < cols - 1:
                width = j - i
                if best_vbar is None or width > best_vbar[2]:
                    best_vbar = (i, j - 1, width, color)
            i = j
        else:
            i += 1

    # --- horizontal bar (pure rows) ---
    row_color = []
    for r in range(rows):
        vals = set(input_grid[r][c] for c in range(cols))
        row_color.append(vals.pop() if len(vals) == 1 else None)

    best_hbar = None
    i = 0
    while i < rows:
        if row_color[i] is not None:
            color = row_color[i]
            j = i
            while j < rows and row_color[j] == color:
                j += 1
            if i > 0 and j - 1 < rows - 1:
                height = j - i
                if best_hbar is None or height > best_hbar[2]:
                    best_hbar = (i, j - 1, height, color)
            i = j
        else:
            i += 1

    # Choose the best bar found
    if best_vbar and best_hbar:
        if best_vbar[2] >= best_hbar[2]:
            bar_start, bar_end, _, bar_color = best_vbar
            bar_orient = 'vertical'
        else:
            bar_start, bar_end, _, bar_color = best_hbar
            bar_orient = 'horizontal'
    elif best_vbar:
        bar_start, bar_end, _, bar_color = best_vbar
        bar_orient = 'vertical'
    elif best_hbar:
        bar_start, bar_end, _, bar_color = best_hbar
        bar_orient = 'horizontal'
    else:
        return [row[:] for row in input_grid]

    # Collect colors in the non-bar regions to find background and markers
    non_bar_counts: dict[int, int] = {}
    if bar_orient == 'vertical':
        for r in range(rows):
            for c in range(bar_start):
                non_bar_counts[input_grid[r][c]] = non_bar_counts.get(input_grid[r][c], 0) + 1
            for c in range(bar_end + 1, cols):
                non_bar_counts[input_grid[r][c]] = non_bar_counts.get(input_grid[r][c], 0) + 1
    else:
        for r in range(rows):
            for c in range(cols):
                if r < bar_start or r > bar_end:
                    non_bar_counts[input_grid[r][c]] = non_bar_counts.get(input_grid[r][c], 0) + 1

    if not non_bar_counts:
        return [row[:] for row in input_grid]

    bg_color = max(non_bar_counts, key=non_bar_counts.get)
    marker_colors = set(non_bar_counts.keys()) - {bg_color}

    if not marker_colors:
        return [row[:] for row in input_grid]

    result = [row[:] for row in input_grid]

    if bar_orient == 'vertical':
        for r in range(rows):
            left_count = sum(1 for c in range(bar_start)
                             if input_grid[r][c] in marker_colors)
            right_count = sum(1 for c in range(bar_end + 1, cols)
                              if input_grid[r][c] in marker_colors)

            for c in range(max(0, bar_start - left_count), bar_start):
                result[r][c] = bar_color
            for c in range(bar_end + 1, min(cols, bar_end + 1 + right_count)):
                result[r][c] = bar_color

            for c in range(bar_start):
                if result[r][c] in marker_colors:
                    result[r][c] = bg_color
            for c in range(bar_end + 1, cols):
                if result[r][c] in marker_colors:
                    result[r][c] = bg_color

    else:  # horizontal
        for c in range(cols):
            above_count = sum(1 for r in range(bar_start)
                              if input_grid[r][c] in marker_colors)
            below_count = sum(1 for r in range(bar_end + 1, rows)
                              if input_grid[r][c] in marker_colors)

            for r in range(max(0, bar_start - above_count), bar_start):
                result[r][c] = bar_color
            for r in range(bar_end + 1, min(rows, bar_end + 1 + below_count)):
                result[r][c] = bar_color

            for r in range(bar_start):
                if result[r][c] in marker_colors:
                    result[r][c] = bg_color
            for r in range(bar_end + 1, rows):
                if result[r][c] in marker_colors:
                    result[r][c] = bg_color

    return result
