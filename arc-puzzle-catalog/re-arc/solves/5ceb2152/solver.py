def transform(input_grid):
    import copy
    from collections import Counter

    grid = copy.deepcopy(input_grid)
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg_color = color_counts.most_common(1)[0][0]

    # Find foreground color
    fg_color = [c for c in color_counts if c != bg_color][0]

    # Collect all foreground cells
    fg_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == fg_color:
                fg_cells.add((r, c))

    # Get unique columns and split into two groups by largest gap
    all_cols = sorted(set(c for _, c in fg_cells))
    max_gap = -1
    split_idx = 0
    for i in range(len(all_cols) - 1):
        gap = all_cols[i + 1] - all_cols[i]
        if gap > max_gap:
            max_gap = gap
            split_idx = i

    left_col_set = set(all_cols[:split_idx + 1])
    right_col_set = set(all_cols[split_idx + 1:])

    left_cells = [(r, c) for r, c in fg_cells if c in left_col_set]
    right_cells = [(r, c) for r, c in fg_cells if c in right_col_set]

    left_max_row = max(r for r, _ in left_cells)
    right_max_row = max(r for r, _ in right_cells)
    left_anchor = min(left_col_set)
    right_anchor = min(right_col_set)

    # Build row -> list of column offsets
    def build_row_offsets(cells, anchor):
        mapping = {}
        for r, c in cells:
            mapping.setdefault(r, []).append(c - anchor)
        return mapping

    if left_max_row < right_max_row:
        longer_offsets = build_row_offsets(right_cells, right_anchor)
        shorter_anchor = left_anchor
        ext_start, ext_end = left_max_row + 1, right_max_row
    elif right_max_row < left_max_row:
        longer_offsets = build_row_offsets(left_cells, left_anchor)
        shorter_anchor = right_anchor
        ext_start, ext_end = right_max_row + 1, left_max_row
    else:
        return grid

    for r in range(ext_start, ext_end + 1):
        if r in longer_offsets:
            for offset in longer_offsets[r]:
                new_c = shorter_anchor + offset
                if 0 <= new_c < cols:
                    grid[r][new_c] = 9

    return grid
