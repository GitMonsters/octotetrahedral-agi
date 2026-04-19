def transform(grid):
    from collections import Counter, deque
    
    H = len(grid)
    W = len(grid[0])
    
    counts = Counter()
    for row in grid:
        for v in row:
            counts[v] += 1
    bg = counts.most_common(1)[0][0]
    
    # Find connected components
    in_cells = {}
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                in_cells[(r, c)] = grid[r][c]
    
    if not in_cells:
        return [row[:] for row in grid[:H - 1]]
    
    visited = set()
    components = []
    for r in range(H):
        for c in range(W):
            if (r, c) in in_cells and (r, c) not in visited:
                comp = {}
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp[(cr, cc)] = in_cells[(cr, cc)]
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr, nc) in in_cells and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)
    
    components.sort(key=lambda c: (min(r for r,_ in c), min(c_ for _,c_ in c)))
    
    # Find connector color
    comp_colors = []
    for comp in components:
        colors = set(comp.values())
        if len(colors) > 1:
            comp_colors.append(colors)
    
    conn_color = None
    if comp_colors:
        common = comp_colors[0]
        for s in comp_colors[1:]:
            common = common & s
        if len(common) == 1:
            conn_color = list(common)[0]
        elif len(common) > 1:
            best = None
            best_score = -1
            for cc in common:
                score = sum(1 for comp in components
                           if cc in set(comp.values()) and
                           Counter(comp.values())[cc] <= len(comp) // 2)
                if score > best_score:
                    best_score = score
                    best = cc
            conn_color = best
    
    if conn_color is not None:
        return _solve_with_connectors(grid, components, conn_color, bg, H, W)
    else:
        return _solve_without_connectors(grid, components, bg, H, W)


def _solve_with_connectors(grid, components, conn_color, bg, H, W):
    comp_data = []
    for comp in components:
        colors = set(comp.values())
        if conn_color in colors:
            main_color = [c for c in colors if c != conn_color]
            main_color = main_color[0] if main_color else conn_color
        else:
            main_color = list(colors)[0]
        
        connectors = [(r, c) for (r, c), v in comp.items() if v == conn_color]
        connectors.sort(key=lambda p: (p[0], p[1]))
        
        converted = {}
        for (r, c), v in comp.items():
            converted[(r, c)] = main_color if v == conn_color else v
        
        entry = connectors[0] if connectors else None
        exit_pt = connectors[-1] if connectors else None
        if len(connectors) == 1:
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            cr, cc = connectors[0]
            if cr - min_r <= max_r - cr:
                entry = connectors[0]
                exit_pt = None
            else:
                entry = None
                exit_pt = connectors[0]
        
        comp_data.append({
            'cells': converted,
            'main_color': main_color,
            'entry': entry,
            'exit': exit_pt,
        })
    
    placed_cells = {}
    first = comp_data[0]
    for (r, c), v in first['cells'].items():
        placed_cells[(r, c)] = v
    
    prev_exit = first['exit']
    if prev_exit is None:
        max_r = max(r for r, c in first['cells'])
        bottom_cells = [(r, c) for r, c in first['cells'] if r == max_r]
        prev_exit = max(bottom_cells, key=lambda p: p[1])
    
    for i in range(1, len(comp_data)):
        cd = comp_data[i]
        entry = cd['entry']
        if entry is None:
            min_r = min(r for r, c in cd['cells'])
            top_cells = [(r, c) for r, c in cd['cells'] if r == min_r]
            entry = min(top_cells, key=lambda p: p[1])
        
        pr, pc = prev_exit
        er, ec = entry
        dr = pr + 1 - er
        dc = pc - ec
        
        for (r, c), v in cd['cells'].items():
            placed_cells[(r + dr, c + dc)] = v
        
        if cd['exit'] is not None:
            er2, ec2 = cd['exit']
            prev_exit = (er2 + dr, ec2 + dc)
        else:
            max_r = max(r + dr for r, c in cd['cells'])
            bottom_cells = [(r + dr, c + dc) for r, c in cd['cells'] if r + dr == max_r]
            prev_exit = max(bottom_cells, key=lambda p: p[1])
    
    if not placed_cells:
        return [row[:] for row in grid[:H - 1]]
    
    min_r = min(r for r, c in placed_cells)
    max_r = max(r for r, c in placed_cells)
    first_comp_min_r = min(r for r, c in comp_data[0]['cells'])
    out_start = min(first_comp_min_r, min_r)
    if out_start < 0:
        out_start = 0
    out_H = max_r + 1
    
    out = []
    for r in range(out_start, out_H):
        row = [bg] * W
        for c in range(W):
            if (r, c) in placed_cells:
                row[c] = placed_cells[(r, c)]
        out.append(row)
    return out


def _solve_without_connectors(grid, components, bg, H, W):
    if not components:
        return [row[:] for row in grid[:H - 1]]
    
    # Find single-cell "waypoint" components and multi-cell components
    waypoints = [c for c in components if len(c) == 1]
    multi = [c for c in components if len(c) > 1]
    
    if not multi:
        # Only waypoints or nothing useful - just remove gap rows
        return [row[:] for row in grid[:H - 1]]
    
    color = list(multi[0].values())[0]
    
    if not waypoints:
        return [row[:] for row in grid[:H - 1]]
    
    # Reverse order: largest multi-cell first (was at bottom in input)
    multi.sort(key=lambda c: -len(c))
    
    # LR-reflect the first (largest) component
    def lr_reflect(comp):
        cells = list(comp.keys())
        cs = [c for r, c in cells]
        min_c, max_c = min(cs), max(cs)
        return {(r, max_c + min_c - c): v for (r, c), v in comp.items()}
    
    # Fix corners in reflected stair (swap missing cell at 2x2 turns)
    def fix_stair_corners(cells_set):
        cells_set = set(cells_set)
        changed = True
        max_iters = 1
        while changed and max_iters > 0:
            changed = False
            max_iters -= 1
            for (r, c) in sorted(list(cells_set)):
                # Check all 2x2 squares containing this cell
                for dr, dc in [(0,0), (0,-1), (-1,0), (-1,-1)]:
                    square = [(r+dr, c+dc), (r+dr, c+dc+1), (r+dr+1, c+dc), (r+dr+1, c+dc+1)]
                    present = [s for s in square if s in cells_set]
                    if len(present) == 3:
                        missing = [s for s in square if s not in cells_set][0]
                        # Check if this is at the narrow end
                        mr, mc = missing
                        # The narrow end has the cell with fewer neighbors
                        diag = (square[0] if missing == square[3] else
                                square[3] if missing == square[0] else
                                square[1] if missing == square[2] else square[2])
                        if diag in cells_set:
                            # Count neighbors for the diagonal cell
                            diag_nbrs = sum(1 for ddr,ddc in [(-1,0),(1,0),(0,-1),(0,1)]
                                          if (diag[0]+ddr, diag[1]+ddc) in cells_set)
                            # If diagonal has 2 neighbors (it's the "bridge" cell at a turn)
                            # And adding the missing cell would create a smoother path
                            # Check: is the missing cell's row the "narrower" side?
                            row_missing = sum(1 for (rr,cc) in cells_set if rr == mr)
                            row_diag = sum(1 for (rr,cc) in cells_set if rr == diag[0])
                            if row_diag > row_missing and diag_nbrs == 2:
                                cells_set.remove(diag)
                                cells_set.add(missing)
                                changed = True
                                break
                if changed:
                    break
        return cells_set
    
    first_comp = multi[0]
    reflected = lr_reflect(first_comp)
    reflected_cells = set(reflected.keys())
    
    # Fix stair corners
    fixed_cells = fix_stair_corners(reflected_cells)
    
    # Find endpoints of fixed shape
    def find_endpoints(cells_set):
        eps = []
        for (r, c) in cells_set:
            nbrs = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      if (r+dr, c+dc) in cells_set)
            if nbrs == 1:
                eps.append((r, c))
        return eps
    
    # Normalize the fixed shape
    min_r = min(r for r, c in fixed_cells)
    min_c = min(c for r, c in fixed_cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in fixed_cells}
    
    # Find exit (bottom endpoint)
    endpoints = find_endpoints(norm_cells)
    if not endpoints:
        endpoints = list(norm_cells)
    
    top_ep = min(endpoints, key=lambda p: (p[0], p[1]))
    bot_ep = max(endpoints, key=lambda p: (p[0], -p[1]))
    
    # Determine starting position: compute total output height
    # Output height = input height - (inter-component bg rows + trailing bg rows + waypoint rows)
    comp_rows = set()
    for comp in components:
        for r, c in comp:
            comp_rows.add(r)
    all_rows = sorted(comp_rows)
    first_non_bg = all_rows[0]
    last_non_bg = all_rows[-1]
    
    gap_rows = 0
    for r in range(first_non_bg, last_non_bg + 1):
        if r not in comp_rows:
            gap_rows += 1
    trailing = H - 1 - last_non_bg
    waypoint_rows = len(waypoints)
    
    out_H = H - gap_rows - trailing - waypoint_rows
    
    # Compute assembly
    # Place the reflected component, waypoint, and other components
    # The reflected component's exit connects to waypoint, which connects to next component
    
    # Build the chain
    placed = {}
    
    # First: place reflected+fixed component at a specific row
    # The component should be positioned so that the full chain fits in out_H rows
    # starting with leading bg rows from the input
    
    # Calculate chain height
    chain_height = max(r for r, c in norm_cells) + 1  # reflected comp height
    chain_height += 1  # waypoint
    for mc in multi[1:]:
        rs = [r for r, c in mc]
        chain_height += max(rs) - min(rs) + 1
    
    # Add bridge rows: estimate based on stair width pattern
    # For now: bridges between waypoint and each multi-comp add width-based rows
    # Simple heuristic: 2 bridge rows for the connection below waypoint
    bridge_rows = 2  # approximate
    chain_height += bridge_rows
    
    leading_bg = out_H - chain_height
    if leading_bg < 0:
        leading_bg = 0
    
    start_row = leading_bg
    
    # Column offset: place reflected shape flush to right wall of grid
    c0 = W - 1 - max(c for r, c in norm_cells)
    
    # Place reflected component
    for (r, c) in norm_cells:
        placed[(r + start_row, c + c0)] = color
    
    # Find exit of reflected component (bottom endpoint)
    exit_r = max(r for r, c in norm_cells) + start_row
    # Exit column: the leftmost cell of the bottom row
    bottom_row_cells = [(r, c) for r, c in norm_cells if r == max(rr for rr, _ in norm_cells)]
    exit_c = min(c for r, c in bottom_row_cells) + c0
    
    # Place bridge cell connecting reflected comp to waypoint
    placed[(exit_r, exit_c - 1)] = color  # one cell left of exit
    
    # Place waypoint one row below
    wp_r = exit_r + 1
    wp_c = exit_c - 1
    placed[(wp_r, wp_c)] = color
    
    # Place bridge from waypoint to next component
    # Use stair pattern: go down-left with width 2
    if len(multi) > 1:
        next_comp = multi[1]
        next_comp_height = max(r for r, c in next_comp) - min(r for r, c in next_comp) + 1
        next_comp_width = max(c for r, c in next_comp) - min(c for r, c in next_comp) + 1
        
        # Target: place next comp so it ends at the bottom of the output
        target_bottom = out_H - 1
        target_top = target_bottom - next_comp_height + 1
        
        # Bridge from waypoint to target_top
        bridge_start_r = wp_r + 1
        bridge_end_r = target_top - 1
        
        for br in range(bridge_start_r, bridge_end_r + 1):
            placed[(br, wp_c - 1)] = color
            placed[(br, wp_c)] = color
        
        # Place next component
        next_cells = list(next_comp.keys())
        next_min_r = min(r for r, c in next_cells)
        next_min_c = min(c for r, c in next_cells)
        
        # Find entry point of next comp (top-left)
        next_entry_c = wp_c  # align with waypoint column
        
        for (r, c), v in next_comp.items():
            nr = r - next_min_r + target_top
            nc = c - next_min_c + next_entry_c
            placed[(nr, nc)] = color
    
    # Build output grid
    out = []
    for r in range(out_H):
        row = [bg] * W
        for c in range(W):
            if (r, c) in placed:
                row[c] = placed[(r, c)]
        out.append(row)
    
    return out
