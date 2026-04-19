def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Find separator rows (all same value)
    sep_rows = set()
    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_rows.add(r)
            if sep_color is None:
                sep_color = grid[r][0]

    if not sep_rows:
        return grid

    # Find separator columns
    non_sep_rows = [r for r in range(rows) if r not in sep_rows]
    sep_cols = set()
    for c in range(cols):
        if all(grid[r][c] == sep_color for r in non_sep_rows):
            sep_cols.add(c)

    # Cell row/col groups
    cell_row_groups = []
    group = []
    for r in range(rows):
        if r in sep_rows:
            if group:
                cell_row_groups.append(group)
                group = []
        else:
            group.append(r)
    if group:
        cell_row_groups.append(group)

    cell_col_groups = []
    group = []
    for c in range(cols):
        if c in sep_cols:
            if group:
                cell_col_groups.append(group)
                group = []
        else:
            group.append(c)
    if group:
        cell_col_groups.append(group)

    # Extract cell grid
    nrows_c = len(cell_row_groups)
    ncols_c = len(cell_col_groups)
    cell_grid = []
    for rg in cell_row_groups:
        row = []
        for cg in cell_col_groups:
            row.append(grid[rg[0]][cg[0]])
        cell_grid.append(row)

    # Background color (most frequent in cell grid)
    flat = [v for row in cell_grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Connected components (4-connected, non-bg)
    visited = [[False] * ncols_c for _ in range(nrows_c)]
    components = []
    for r in range(nrows_c):
        for c in range(ncols_c):
            if not visited[r][c] and cell_grid[r][c] != bg:
                comp = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc, cell_grid[cr][cc]))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < nrows_c and 0 <= nc < ncols_c and not visited[nr][nc] and cell_grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    if not components:
        return grid

    # Template = component with most distinct colors (tiebreak: most cells)
    template = max(components, key=lambda c: (len(set(v for _, _, v in c)), len(c)))
    template_colors = set(v for _, _, v in template)
    template_set = set((r, c) for r, c, v in template)

    # Anchor color: template color also in other components
    other_colors = set()
    for comp in components:
        if set((r, c) for r, c, v in comp) != template_set:
            for _, _, v in comp:
                other_colors.add(v)

    anchor_candidates = template_colors & other_colors
    if not anchor_candidates:
        return grid
    anchor_color = anchor_candidates.pop()

    # Center: first (topmost, leftmost) anchor cell in template
    center = min((r, c) for r, c, v in template if v == anchor_color)

    # Offsets (entire template relative to center)
    offsets = [(r - center[0], c - center[1], v) for r, c, v in template]

    # Stamp at each anchor cell OUTSIDE template
    result = [row[:] for row in cell_grid]
    for r in range(nrows_c):
        for c in range(ncols_c):
            if cell_grid[r][c] == anchor_color and (r, c) not in template_set:
                for dr, dc, v in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows_c and 0 <= nc < ncols_c:
                        result[nr][nc] = v

    # Convert back to pixel grid
    out = [row[:] for row in grid]
    for ci, rg in enumerate(cell_row_groups):
        for cj, cg in enumerate(cell_col_groups):
            val = result[ci][cj]
            for r in rg:
                for c in cg:
                    out[r][c] = val

    return out
