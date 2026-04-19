def transform(input_grid):
    import copy
    from collections import deque
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Detect grid structure
    sep_val = None
    sep_rows = set()
    for r in range(rows):
        if len(set(input_grid[r])) == 1:
            sep_rows.add(r)
            sep_val = input_grid[r][0]
    
    data_rows_indices = [r for r in range(rows) if r not in sep_rows]
    sep_cols = set()
    for c in range(cols):
        if all(input_grid[r][c] == sep_val for r in data_rows_indices):
            sep_cols.add(c)
    
    # Row groups and col groups
    row_groups = []
    group = []
    for r in range(rows):
        if r in sep_rows:
            if group: row_groups.append(tuple(group)); group = []
        else: group.append(r)
    if group: row_groups.append(tuple(group))
    
    col_groups = []
    group = []
    for c in range(cols):
        if c in sep_cols:
            if group: col_groups.append(tuple(group)); group = []
        else: group.append(c)
    if group: col_groups.append(tuple(group))
    
    n_rows = len(row_groups)
    n_cols = len(col_groups)
    
    # Find default cell value
    from collections import Counter
    cell_vals = Counter()
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = input_grid[row_groups[ri][0]][col_groups[ci][0]]
            cell_vals[v] += 1
    default_val = cell_vals.most_common(1)[0][0]
    
    HOLE = 'HOLE'
    
    def get_cell(ri, ci):
        v = input_grid[row_groups[ri][0]][col_groups[ci][0]]
        return HOLE if v == sep_val else v
    
    # Find all non-default cells
    special = {}
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = get_cell(ri, ci)
            if v != default_val:
                special[(ri, ci)] = v
    
    # Find connected components (8-connectivity)
    visited = set()
    components = []
    for cell in special:
        if cell in visited: continue
        comp = set()
        q = deque([cell])
        while q:
            cr, cc = q.popleft()
            if (cr, cc) in visited: continue
            visited.add((cr, cc))
            comp.add((cr, cc))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0: continue
                    nc = (cr+dr, cc+dc)
                    if nc in special and nc not in visited:
                        q.append(nc)
        components.append(comp)
    
    # Largest component = complete pattern
    components.sort(key=len, reverse=True)
    main_comp = components[0]
    other_comps = components[1:]
    
    # Build template from main component
    # Find the hole in main comp (if any) as anchor
    holes_in_main = [(r, c) for r, c in main_comp if special[(r,c)] == HOLE]
    if holes_in_main:
        anchor = holes_in_main[0]
    else:
        anchor = min(main_comp)
    
    template = {}
    for r, c in main_comp:
        dr, dc = r - anchor[0], c - anchor[1]
        template[(dr, dc)] = special[(r, c)]
    
    # Apply template to each incomplete component
    output = copy.deepcopy(input_grid)
    
    def set_cell(ri, ci, val):
        if 0 <= ri < n_rows and 0 <= ci < n_cols:
            for r in row_groups[ri]:
                for c in col_groups[ci]:
                    output[r][c] = sep_val if val == HOLE else val
    
    for comp in other_comps:
        # Try to match comp against template elements
        comp_cells = {(r, c): special[(r, c)] for r, c in comp}
        
        best_offset = None
        for (cr, cc), cv in comp_cells.items():
            for (dr, dc), tv in template.items():
                if cv != tv: continue
                # Proposed anchor at (cr - dr, cc - dc)
                ar, ac = cr - dr, cc - dc
                # Check all comp cells match at this offset
                match = True
                for (cr2, cc2), cv2 in comp_cells.items():
                    tdr, tdc = cr2 - ar, cc2 - ac
                    if (tdr, tdc) not in template or template[(tdr, tdc)] != cv2:
                        match = False
                        break
                if match:
                    best_offset = (ar, ac)
                    break
            if best_offset: break
        
        if best_offset:
            ar, ac = best_offset
            for (dr, dc), tv in template.items():
                ri, ci = ar + dr, ac + dc
                set_cell(ri, ci, tv)
    
    return output
