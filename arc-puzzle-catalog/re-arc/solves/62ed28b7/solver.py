def transform(input_grid):
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    bg = max(color_count, key=color_count.get)

    # Find divider rows: entirely one value that isn't background
    divider_color = None
    divider_rows = set()
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            divider_rows.add(r)
            divider_color = grid[r][0]

    # Find divider columns: entirely the divider color
    divider_cols = set()
    if divider_color is not None:
        for c in range(cols):
            if all(grid[r][c] == divider_color for r in range(rows)):
                divider_cols.add(c)

    # Build block ranges
    if divider_rows or divider_cols:
        block_row_ranges = []
        start = None
        for r in range(rows):
            if r not in divider_rows:
                if start is None:
                    start = r
            else:
                if start is not None:
                    block_row_ranges.append((start, r - 1))
                    start = None
        if start is not None:
            block_row_ranges.append((start, rows - 1))

        block_col_ranges = []
        start = None
        for c in range(cols):
            if c not in divider_cols:
                if start is None:
                    start = c
            else:
                if start is not None:
                    block_col_ranges.append((start, c - 1))
                    start = None
        if start is not None:
            block_col_ranges.append((start, cols - 1))
    else:
        # No grid lines — each cell is its own 1×1 block
        block_row_ranges = [(r, r) for r in range(rows)]
        block_col_ranges = [(c, c) for c in range(cols)]

    # Extract markers: (color, cell_row, cell_col) -> set of (block_row_idx, block_col_idx)
    markers = {}
    for br_idx, (r_start, r_end) in enumerate(block_row_ranges):
        for bc_idx, (c_start, c_end) in enumerate(block_col_ranges):
            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    v = grid[r][c]
                    if v != bg:
                        key = (v, r - r_start, c - c_start)
                        if key not in markers:
                            markers[key] = set()
                        markers[key].add((br_idx, bc_idx))

    # Fill between aligned markers
    output = [row[:] for row in grid]

    for (color, cell_r, cell_c), block_positions in markers.items():
        # Group by block row — fill horizontal gaps
        by_row = {}
        for br, bc in block_positions:
            by_row.setdefault(br, []).append(bc)
        for br, bcs in by_row.items():
            if len(bcs) >= 2:
                for bc in range(min(bcs), max(bcs) + 1):
                    rs, re = block_row_ranges[br]
                    cs, ce = block_col_ranges[bc]
                    ar, ac = rs + cell_r, cs + cell_c
                    if ar <= re and ac <= ce:
                        output[ar][ac] = color

        # Group by block col — fill vertical gaps
        by_col = {}
        for br, bc in block_positions:
            by_col.setdefault(bc, []).append(br)
        for bc, brs in by_col.items():
            if len(brs) >= 2:
                for br in range(min(brs), max(brs) + 1):
                    rs, re = block_row_ranges[br]
                    cs, ce = block_col_ranges[bc]
                    ar, ac = rs + cell_r, cs + cell_c
                    if ar <= re and ac <= ce:
                        output[ar][ac] = color

    return output
