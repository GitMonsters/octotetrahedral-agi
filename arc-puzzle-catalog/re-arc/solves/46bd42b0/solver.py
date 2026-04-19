def transform(input_grid):
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color B (color of all-same rows)
    B = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            B = grid[r][0]
            break
    if B is None:
        return grid

    # Find horizontal band: longest contiguous run of all-B rows
    all_b_rows = [r for r in range(rows) if all(grid[r][c] == B for c in range(cols))]
    if not all_b_rows:
        return grid

    runs = []
    start = all_b_rows[0]
    for i in range(1, len(all_b_rows)):
        if all_b_rows[i] != all_b_rows[i - 1] + 1:
            runs.append((start, all_b_rows[i - 1]))
            start = all_b_rows[i]
    runs.append((start, all_b_rows[-1]))
    hr1, hr2 = max(runs, key=lambda x: x[1] - x[0])

    # Find vertical band: for each column, find maximal contiguous B-range around H
    col_ranges = {}
    for c in range(cols):
        top = hr1
        while top > 0 and grid[top - 1][c] == B:
            top -= 1
        bottom = hr2
        while bottom < rows - 1 and grid[bottom + 1][c] == B:
            bottom += 1
        col_ranges[c] = (top, bottom)

    # Group contiguous columns with same range extending beyond H
    best_vc1 = best_vc2 = best_vr1 = best_vr2 = None
    best_area = 0

    c = 0
    while c < cols:
        rng = col_ranges[c]
        if rng == (hr1, hr2):
            c += 1
            continue
        start_c = c
        while c < cols and col_ranges[c] == rng:
            c += 1
        end_c = c - 1
        area = (rng[1] - rng[0] + 1) * (end_c - start_c + 1)
        if area > best_area:
            best_area = area
            best_vc1 = start_c
            best_vc2 = end_c
            best_vr1 = rng[0]
            best_vr2 = rng[1]

    # Compute interior regions
    # Horizontal band: border rows face data (not at grid edge)
    h_int_start = hr1 + (1 if hr1 > 0 else 0)
    h_int_end = hr2 - (1 if hr2 < rows - 1 else 0)

    output = [row[:] for row in grid]

    # Set horizontal band interior rows to all 0
    if h_int_start <= h_int_end:
        for r in range(h_int_start, h_int_end + 1):
            for c_i in range(cols):
                output[r][c_i] = 0

    # Set vertical band interior to 0
    if best_vc1 is not None:
        v_int_col_start = best_vc1 + (1 if best_vc1 > 0 else 0)
        v_int_col_end = best_vc2 - (1 if best_vc2 < cols - 1 else 0)
        v_int_row_start = best_vr1 + (1 if best_vr1 > 0 else 0)
        v_int_row_end = best_vr2 - (1 if best_vr2 < rows - 1 else 0)

        if v_int_row_start <= v_int_row_end and v_int_col_start <= v_int_col_end:
            for r in range(v_int_row_start, v_int_row_end + 1):
                for c_i in range(v_int_col_start, v_int_col_end + 1):
                    output[r][c_i] = 0

    return output
