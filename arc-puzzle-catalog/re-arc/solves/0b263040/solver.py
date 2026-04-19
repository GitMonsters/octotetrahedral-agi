def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]

    # Bounding box of non-background (outer rectangle)
    outer_top = outer_bottom = None
    outer_left, outer_right = cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                if outer_top is None:
                    outer_top = r
                outer_bottom = r
                outer_left = min(outer_left, c)
                outer_right = max(outer_right, c)

    # Frame color = dominant on outer rect edges; inner color = the other
    edge_colors = {}
    for c in range(outer_left, outer_right + 1):
        for r in [outer_top, outer_bottom]:
            v = grid[r][c]
            if v != bg:
                edge_colors[v] = edge_colors.get(v, 0) + 1
    for r in range(outer_top, outer_bottom + 1):
        for c in [outer_left, outer_right]:
            v = grid[r][c]
            if v != bg:
                edge_colors[v] = edge_colors.get(v, 0) + 1
    frame_color = max(edge_colors, key=edge_colors.get)

    inner_color = None
    for r in range(outer_top, outer_bottom + 1):
        for c in range(outer_left, outer_right + 1):
            if grid[r][c] != bg and grid[r][c] != frame_color:
                inner_color = grid[r][c]
                break
        if inner_color is not None:
            break

    # If no inner color found, just return the grid unchanged
    if inner_color is None:
        return [row[:] for row in grid]

    # Inner rectangle: first/last row/col where inner_count >= frame_count
    inner_top = inner_bottom = inner_left = inner_right = None
    for r in range(outer_top, outer_bottom + 1):
        ic = sum(1 for c in range(outer_left, outer_right + 1) if grid[r][c] == inner_color)
        fc = sum(1 for c in range(outer_left, outer_right + 1) if grid[r][c] == frame_color)
        if ic >= fc:
            inner_top = r
            break
    for r in range(outer_bottom, outer_top - 1, -1):
        ic = sum(1 for c in range(outer_left, outer_right + 1) if grid[r][c] == inner_color)
        fc = sum(1 for c in range(outer_left, outer_right + 1) if grid[r][c] == frame_color)
        if ic >= fc:
            inner_bottom = r
            break
    for c in range(outer_left, outer_right + 1):
        ic = sum(1 for r in range(outer_top, outer_bottom + 1) if grid[r][c] == inner_color)
        fc = sum(1 for r in range(outer_top, outer_bottom + 1) if grid[r][c] == frame_color)
        if ic >= fc:
            inner_left = c
            break
    for c in range(outer_right, outer_left - 1, -1):
        ic = sum(1 for r in range(outer_top, outer_bottom + 1) if grid[r][c] == inner_color)
        fc = sum(1 for r in range(outer_top, outer_bottom + 1) if grid[r][c] == frame_color)
        if ic >= fc:
            inner_right = c
            break

    # Signal columns from top/bottom border notches
    signal_cols = set()
    for r in range(outer_top, inner_top):
        for c in range(outer_left, outer_right + 1):
            if grid[r][c] == inner_color:
                signal_cols.add(c)
    for r in range(inner_bottom + 1, outer_bottom + 1):
        for c in range(outer_left, outer_right + 1):
            if grid[r][c] == inner_color:
                signal_cols.add(c)

    # Signal rows from left/right border notches
    signal_rows = set()
    for c in range(outer_left, inner_left):
        for r in range(outer_top, outer_bottom + 1):
            if grid[r][c] == inner_color:
                signal_rows.add(r)
    for c in range(inner_right + 1, outer_right + 1):
        for r in range(outer_top, outer_bottom + 1):
            if grid[r][c] == inner_color:
                signal_rows.add(r)

    # Build output
    output = [row[:] for row in grid]

    # Fill outer rectangle with frame color
    for r in range(outer_top, outer_bottom + 1):
        for c in range(outer_left, outer_right + 1):
            output[r][c] = frame_color

    # Fill interior based on signals
    for r in range(inner_top, inner_bottom + 1):
        for c in range(inner_left, inner_right + 1):
            if r in signal_rows or c in signal_cols:
                output[r][c] = frame_color
            else:
                output[r][c] = inner_color

    # Extend signal columns into background (above and below)
    for c in signal_cols:
        for r in range(0, outer_top):
            output[r][c] = inner_color
        for r in range(outer_bottom + 1, rows):
            output[r][c] = inner_color

    # Extend signal rows into background (left and right)
    for r in signal_rows:
        for c in range(0, outer_left):
            output[r][c] = inner_color
        for c in range(outer_right + 1, cols):
            output[r][c] = inner_color

    return output
