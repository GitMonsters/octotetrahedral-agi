def transform(input_grid):
    """
    The input is a tiled grid with fill_color and sep_color, with marker values at
    grid line intersections. The output is derived from 2x2 blocks of the intersection grid.
    
    For all-same blocks: output = that value.
    For mixed blocks: output = fill_color.
    For all-same sep_color blocks specifically: output depends on the context of the
    row and column group structure.
    """
    from collections import Counter
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find fill and separator colors
    row_patterns = Counter(tuple(row) for row in input_grid)
    most_common_row = row_patterns.most_common(1)[0][0]
    row_counts = Counter(most_common_row)
    fill_color = row_counts.most_common(1)[0][0]
    v_sep_cols = sorted(j for j, v in enumerate(most_common_row) if v != fill_color)
    sep_color = most_common_row[v_sep_cols[0]]
    cell_positions = [j for j, v in enumerate(most_common_row) if v == fill_color]
    
    h_sep_rows = [r for r in range(rows)
                  if all(input_grid[r][c] != fill_color for c in cell_positions[:3])]
    
    int_grid = [[input_grid[r][c] for c in v_sep_cols] for r in h_sep_rows]
    nr = len(int_grid)
    nc = len(int_grid[0])
    
    # Group consecutive identical rows and cols
    def make_groups(n, get_key):
        groups = []
        i = 0
        while i < n:
            j = i + 1
            while j < n and get_key(j) == get_key(i):
                j += 1
            groups.append(list(range(i, j)))
            i = j
        return groups
    
    row_groups = make_groups(nr, lambda r: tuple(int_grid[r]))
    col_groups = make_groups(nc, lambda c: tuple(int_grid[r][c] for r in range(nr)))
    
    def is_all_std_rg(g):
        return all(int_grid[r][c] == sep_color for r in g for c in range(nc))
    def is_all_std_cg(g):
        return all(int_grid[r][c] == sep_color for r in range(nr) for c in g)
    
    # Select indices
    def select_indices(groups, is_std_fn):
        trimmed = list(groups)
        while trimmed and len(trimmed[0]) == 1 and is_std_fn(trimmed[0]):
            trimmed.pop(0)
        while trimmed and len(trimmed[-1]) == 1 and is_std_fn(trimmed[-1]):
            trimmed.pop()
        if (len(trimmed) >= 2 and
            len(trimmed[0]) >= 2 and is_std_fn(trimmed[0]) and
            len(trimmed[-1]) >= 2 and is_std_fn(trimmed[-1])):
            trimmed = trimmed[1:-1]
        indices = []
        for g in trimmed:
            indices.extend(g)
        return indices, trimmed
    
    row_indices, t_rg = select_indices(row_groups, is_all_std_rg)
    col_indices, t_cg = select_indices(col_groups, is_all_std_cg)
    
    out_rows = len(row_indices) - 1
    out_cols = len(col_indices) - 1
    
    # For each row/col index, determine group membership
    def idx_to_group(idx, groups):
        for gi, g in enumerate(groups):
            if idx in g:
                return gi
        return -1
    
    # For handling all-same-sep: determine if a pair is "within same group" and if the group is all-std
    # Also determine the col pair's relationship to the non-std col groups
    
    # Find which col groups are all-std
    std_col_group_indices = set()
    nonstd_col_group_indices = set()
    for gi, g in enumerate(t_cg):
        if is_all_std_cg(g):
            std_col_group_indices.add(gi)
        else:
            nonstd_col_group_indices.add(gi)
    
    # For all-sep blocks in rows where the row group is all-std:
    # The pattern depends on whether the col pair is within a std/nonstd group
    # and whether both int grid columns in the pair are identical
    
    output = []
    for r in range(out_rows):
        row = []
        r0, r1 = row_indices[r], row_indices[r + 1]
        rg0 = idx_to_group(r0, t_rg)
        rg1 = idx_to_group(r1, t_rg)
        r_same_group = (rg0 == rg1)
        r_group_std = r_same_group and is_all_std_rg(t_rg[rg0])
        r_is_transition = not r_same_group
        
        for c in range(out_cols):
            c0, c1 = col_indices[c], col_indices[c + 1]
            cg0 = idx_to_group(c0, t_cg)
            cg1 = idx_to_group(c1, t_cg)
            c_same_group = (cg0 == cg1)
            
            block = [int_grid[r0][c0], int_grid[r0][c1],
                     int_grid[r1][c0], int_grid[r1][c1]]
            
            if len(set(block)) == 1 and block[0] != sep_color:
                row.append(block[0])
            elif len(set(block)) > 1:
                row.append(fill_color)
            else:
                # All same sep_color - determine fill vs sep
                if not c_same_group:
                    # Between different col groups
                    # If both col groups are singletons -> sep, otherwise -> fill
                    if len(t_cg[cg0]) == 1 and len(t_cg[cg1]) == 1:
                        row.append(sep_color)
                    else:
                        row.append(fill_color)
                elif r_group_std:
                    # All-std row group, within same col group
                    # Sep at paired positions (within sub-pairs) and non-std col groups
                    if cg0 in nonstd_col_group_indices:
                        row.append(sep_color)
                    else:
                        # Within std col group: check sub-pair position
                        g = t_cg[cg0]
                        pos0 = g.index(c0)
                        if pos0 % 2 == 0:  # first of a sub-pair
                            row.append(sep_color)
                        else:
                            row.append(fill_color)
                elif r_is_transition:
                    # Transition between row groups
                    both_nonstd = (not is_all_std_rg(t_rg[rg0]) and not is_all_std_rg(t_rg[rg1]))
                    if both_nonstd and c_same_group and cg0 in std_col_group_indices:
                        g = t_cg[cg0]
                        pos0 = g.index(c0)
                        if pos0 % 2 == 1:  # boundary position
                            row.append(sep_color)
                        else:
                            row.append(fill_color)
                    else:
                        row.append(fill_color)
                else:
                    row.append(fill_color)
        output.append(row)
    
    return output
