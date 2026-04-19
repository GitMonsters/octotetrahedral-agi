def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import numpy as np
    grid = np.array(input_grid)
    H, W = grid.shape

    # Find background (most common value)
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Find divider rows: rows where a dominant non-bg color covers >80% of cells
    div_rows = []
    for r in range(H):
        row = grid[r]
        non_bg = row[row != bg]
        if len(non_bg) > W * 0.8:
            vr, cr = np.unique(non_bg, return_counts=True)
            color = int(vr[np.argmax(cr)])
            div_rows.append((r, color))

    # Find divider columns
    div_cols = []
    for c in range(W):
        col = grid[:, c]
        non_bg = col[col != bg]
        if len(non_bg) > H * 0.8:
            vc, cc = np.unique(non_bg, return_counts=True)
            color = int(vc[np.argmax(cc)])
            div_cols.append((c, color))

    div_row_pos = set(r for r, _ in div_rows)
    div_col_pos = set(c for c, _ in div_cols)

    # Find gap rows: rows where all non-divider-col cells are bg
    gap_rows = []
    for r in range(H):
        if r in div_row_pos:
            continue
        is_gap = all(
            grid[r, c] == bg or c in div_col_pos
            for c in range(W)
        )
        if is_gap:
            gap_rows.append(r)

    # Find gap cols: cols where all non-divider-row cells are bg
    gap_cols = []
    for c in range(W):
        if c in div_col_pos:
            continue
        is_gap = all(
            grid[r, c] == bg or r in div_row_pos
            for r in range(H)
        )
        if is_gap:
            gap_cols.append(c)

    # Collect all row boundaries (dividers + gaps)
    row_bounds = [(r, color) for r, color in div_rows]
    col_bounds = [(c, color) for c, color in div_cols]

    # Add gap rows/cols as boundaries (color = bg)
    for r in gap_rows:
        row_bounds.append((r, bg))
    for c in gap_cols:
        col_bounds.append((c, bg))

    row_bounds.sort()
    col_bounds.sort()

    # We need exactly 2 row boundaries and 2 col boundaries for the subgrid.
    # If more than 2, pick the pair that forms the "center" band.
    def pick_center_pair(bounds, total_size):
        if len(bounds) == 2:
            return bounds[0], bounds[1]
        if len(bounds) > 2:
            # Pick the two boundaries that create the middle band
            # For N boundaries, the middle pair is at indices (N//2-1, N//2)
            n = len(bounds)
            return bounds[n // 2 - 1], bounds[n // 2]
        # len(bounds) == 1: use the single boundary + closer grid edge
        b = bounds[0]
        dist_to_start = b[0]
        dist_to_end = total_size - 1 - b[0]
        if dist_to_end <= dist_to_start:
            # Closer to end, extend to end
            return b, (total_size - 1, bg)
        else:
            return (0, bg), b

    # Filter gap rows/cols to only keep meaningful ones (adjacent to dividers)
    # For gap rows: keep only those where divider cols have breaks
    meaningful_gap_rows = []
    for r in gap_rows:
        # Check if any divider column has a break (bg value) at this row
        has_break = any(grid[r, c] == bg for c, _ in div_cols)
        if has_break:
            meaningful_gap_rows.append(r)

    meaningful_gap_cols = []
    for c in gap_cols:
        has_break = any(grid[r, c] == bg for r, _ in div_rows)
        if has_break:
            meaningful_gap_cols.append(c)

    # Rebuild bounds using only dividers + meaningful gaps
    row_bounds = [(r, color) for r, color in div_rows]
    for r in meaningful_gap_rows:
        row_bounds.append((r, bg))
    row_bounds.sort()

    col_bounds = [(c, color) for c, color in div_cols]
    for c in meaningful_gap_cols:
        col_bounds.append((c, bg))
    col_bounds.sort()

    # Pick boundaries
    rb1, rb2 = pick_center_pair(row_bounds, H)
    cb1, cb2 = pick_center_pair(col_bounds, W)

    r1, r1_color = rb1
    r2, r2_color = rb2
    c1, c1_color = cb1
    c2, c2_color = cb2

    # Extract subgrid
    subgrid = grid[r1:r2 + 1, c1:c2 + 1].copy()
    sh, sw = subgrid.shape

    # Identify mark color in the interior
    # Interior = cells not on the border rows/cols
    interior = subgrid[1:sh - 1, 1:sw - 1]
    interior_non_bg = interior[interior != bg]
    if len(interior_non_bg) == 0:
        # No marks, return subgrid as-is
        return subgrid.tolist()

    mark_vals, mark_counts = np.unique(interior_non_bg, return_counts=True)
    mark_color = int(mark_vals[np.argmax(mark_counts)])

    # Determine fill direction based on which border matches the mark color
    border_colors = {
        'top': r1_color,
        'bottom': r2_color,
        'left': c1_color,
        'right': c2_color,
    }

    matching_borders = [k for k, v in border_colors.items() if v == mark_color]

    # Apply fill
    if 'left' in matching_borders or 'right' in matching_borders:
        # Horizontal fill
        if 'left' in matching_borders:
            # Fill from left to rightmost mark per row
            for r in range(1, sh - 1):
                # Find rightmost mark in this row's interior
                rightmost = -1
                for c in range(1, sw - 1):
                    if subgrid[r, c] == mark_color:
                        rightmost = c
                if rightmost > 0:
                    for c in range(1, rightmost + 1):
                        if subgrid[r, c] == bg:
                            subgrid[r, c] = mark_color
        elif 'right' in matching_borders:
            # Fill from right to leftmost mark per row
            for r in range(1, sh - 1):
                leftmost = sw
                for c in range(1, sw - 1):
                    if subgrid[r, c] == mark_color:
                        leftmost = c
                        break
                if leftmost < sw:
                    for c in range(leftmost, sw - 1):
                        if subgrid[r, c] == bg:
                            subgrid[r, c] = mark_color

    elif 'bottom' in matching_borders or 'top' in matching_borders:
        # Vertical fill
        if 'bottom' in matching_borders:
            # Fill from bottom to topmost mark per column
            for c in range(1, sw - 1):
                topmost = sh
                for r in range(1, sh - 1):
                    if subgrid[r, c] == mark_color:
                        topmost = r
                        break
                if topmost < sh:
                    for r in range(topmost, sh - 1):
                        if subgrid[r, c] == bg:
                            subgrid[r, c] = mark_color
        elif 'top' in matching_borders:
            # Fill from top to bottommost mark per column
            for c in range(1, sw - 1):
                bottommost = -1
                for r in range(1, sh - 1):
                    if subgrid[r, c] == mark_color:
                        bottommost = r
                if bottommost >= 0:
                    for r in range(1, bottommost + 1):
                        if subgrid[r, c] == bg:
                            subgrid[r, c] = mark_color

    return subgrid.tolist()
