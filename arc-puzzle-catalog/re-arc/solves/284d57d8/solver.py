from collections import deque


def transform(input_grid):
    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def get_4conn_components(g):
        visited = set()
        components = []
        for r in range(H):
            for c in range(W):
                if g[r][c] != bg and (r, c) not in visited:
                    color = g[r][c]
                    comp = set()
                    q = deque([(r, c)])
                    visited.add((r, c))
                    while q:
                        cr, cc = q.popleft()
                        comp.add((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and g[nr][nc] == color:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                    components.append((color, comp))
        return components

    def get_8conn_components(cells):
        """Get 8-connected components from a set of cells."""
        remaining = set(cells)
        comps = []
        while remaining:
            start = next(iter(remaining))
            comp = set()
            q = deque([start])
            remaining.discard(start)
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in remaining:
                            remaining.discard((nr, nc))
                            q.append((nr, nc))
            comps.append(comp)
        return comps

    def is_4adj(r, c, cells):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in cells:
                return True
        return False

    def count_8neighbors_in(r, c, cells):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in cells:
                    count += 1
        return count

    def line_reflect(indicator_rc, adj_cell, shape_cells):
        ir, ic = indicator_rc
        sr, sc = adj_cell
        reflected = set()
        for cr, cc in shape_cells:
            if ir == sr:
                nr, nc = cr, ic + sc - cc
            else:
                nr, nc = ir + sr - cr, cc
            if in_bounds(nr, nc):
                reflected.add((nr, nc))
        return reflected

    def line_reflect_h(cells, ic, ac):
        """Reflect cells through vertical line at (ic+ac)/2."""
        return {(r, ic + ac - c) for r, c in cells if in_bounds(r, ic + ac - c)}

    def line_reflect_v(cells, ir, ar):
        """Reflect cells through horizontal line at (ir+ar)/2."""
        return {(ir + ar - r, c) for r, c in cells if in_bounds(ir + ar - r, c)}

    def point_reflect(indicator_rc, adj_cell, shape_cells):
        ir, ic = indicator_rc
        sr, sc = adj_cell
        reflected = set()
        for cr, cc in shape_cells:
            nr, nc = ir + sr - cr, ic + sc - cc
            if in_bounds(nr, nc) and not is_4adj(nr, nc, shape_cells):
                reflected.add((nr, nc))
        return reflected

    def score_bg(cells, indicator_rc):
        return sum(1 for r, c in cells if (r, c) != indicator_rc and grid[r][c] == bg)

    def has_away(indicator_rc, adj_cell, shape_cells):
        ir, ic = indicator_rc
        ar, ac = adj_cell
        return (ar + ar - ir, ac + ac - ic) in shape_cells

    def cardinal_singleton_filter(reflected, shape_cells, indicator_rc):
        """Remove reflected cells that are isolated in the reflected set
        (only neighbor in reflected set is the indicator position)."""
        filtered = set()
        for r, c in reflected:
            if (r, c) == indicator_rc:
                filtered.add((r, c))
                continue
            has_reflected_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in reflected and (nr, nc) != indicator_rc:
                    has_reflected_neighbor = True
                    break
            if has_reflected_neighbor:
                filtered.add((r, c))
        return filtered

    def do_reflection(s_col, s_rc, m_col, m_cells, new_cells):
        """Process one indicator reflecting one shape."""
        sr, sc = s_rc
        card_adjs = []
        diag_adjs = []
        for ddr in [-1, 0, 1]:
            for ddc in [-1, 0, 1]:
                if ddr == 0 and ddc == 0:
                    continue
                nr, nc = sr + ddr, sc + ddc
                if (nr, nc) in m_cells:
                    if ddr == 0 or ddc == 0:
                        card_adjs.append((nr, nc))
                    else:
                        diag_adjs.append((nr, nc))

        if not card_adjs and not diag_adjs:
            return

        both = bool(card_adjs) and bool(diag_adjs)
        two_cards = len(card_adjs) >= 2

        # Best cardinal
        best_ca = None
        best_cs = -1
        for adj in card_adjs:
            ref = line_reflect(s_rc, adj, m_cells)
            s = score_bg(ref, s_rc)
            if s > best_cs:
                best_cs = s
                best_ca = adj

        # Best diagonal
        best_da = None
        best_ds = -1
        best_dr = None
        for adj in diag_adjs:
            ref = point_reflect(s_rc, adj, m_cells)
            s = score_bg(ref, s_rc)
            if s > best_ds:
                best_ds = s
                best_da = adj
                best_dr = ref

        # Choose main reflection type
        use_diag = False
        if both and best_ca:
            is_horiz = (sr == best_ca[0])
            if is_horiz and not has_away(s_rc, best_ca, m_cells):
                use_diag = True

        # For two cardinals: choose based on shape neighbor count
        main_ca = best_ca
        back_ca = None
        if two_cards and not both:
            # Pick cardinal adj with more shape neighbors for main
            scored = []
            for adj in card_adjs:
                n = count_8neighbors_in(adj[0], adj[1], m_cells)
                scored.append((n, adj))
            scored.sort(reverse=True)
            main_ca = scored[0][1]
            back_ca = scored[1][1] if len(scored) > 1 else None

        if use_diag and best_dr and best_ds > 0:
            main_cells = best_dr
            main_type = 'diag'
        elif main_ca:
            # For cardinal, exclude shape cells 4-adj to indicator (except adj cell)
            filt = {c for c in m_cells if c == main_ca or not is_4adj(c[0], c[1], {s_rc})}
            main_cells = line_reflect(s_rc, main_ca, filt)
            main_type = 'card'
        elif best_dr:
            main_cells = best_dr
            main_type = 'diag'
        else:
            return

        # Case 2: cardinal + 2+ diag adjs -> back through diagonal
        # When back-reflecting, exclude main cells whose source's back position
        # is already in the original shape
        exclude_from_main = set()
        if not use_diag and both and len(diag_adjs) >= 2:
            best_br = None
            best_br_score = -1
            best_br_dadj = None
            for dadj in diag_adjs:
                ir2, ic2 = s_rc
                br = line_reflect_v(m_cells, ir2, dadj[0])
                br = {(r, c) for r, c in br if (r, c) not in m_cells}
                brs = sum(1 for r, c in br if grid[r][c] == bg)
                if brs > best_br_score:
                    best_br_score = brs
                    best_br = br
                    best_br_dadj = dadj
            if best_br and best_br_dadj:
                for r, c in best_br:
                    if grid[r][c] == bg and (r, c) not in new_cells:
                        new_cells[(r, c)] = m_col
                # Compute exclusions: shape cells whose back position is in shape
                ir2, ic2 = s_rc
                for cr, cc in m_cells:
                    back_r = ir2 + best_br_dadj[0] - cr
                    if in_bounds(back_r, cc) and (back_r, cc) in m_cells:
                        # Exclude the main reflected position of this cell
                        if ir2 == main_ca[0]:
                            excl_pos = (cr, ic2 + main_ca[1] - cc)
                        else:
                            excl_pos = (ir2 + main_ca[0] - cr, cc)
                        exclude_from_main.add(excl_pos)

        # Paint main reflection with indicator color
        for r, c in main_cells:
            if (r, c) != s_rc and grid[r][c] == bg and (r, c) not in exclude_from_main:
                if (r, c) not in new_cells:
                    new_cells[(r, c)] = s_col

        # Back-reflection logic
        # Case 1: diagonal chosen + cardinal exists -> back through cardinal
        if use_diag and both and best_ca:
            ir2, ic2 = s_rc
            ar, ac = best_ca
            if ir2 == ar:
                back_cells = line_reflect_h(main_cells, ic2, ac)
            else:
                back_cells = line_reflect_v(main_cells, ir2, ar)
            for r, c in back_cells:
                if (r, c) not in m_cells and grid[r][c] == bg and (r, c) not in new_cells:
                    new_cells[(r, c)] = m_col

        # Case 2 back-reflection already handled above (before main paint)

        # Case 3: two cardinal adjs -> back through the other cardinal
        if two_cards and not both and back_ca:
            back_cells = line_reflect(s_rc, back_ca, main_cells)
            for r, c in back_cells:
                if (r, c) not in m_cells and grid[r][c] == bg and (r, c) not in new_cells:
                    new_cells[(r, c)] = m_col

    max_waves = 20
    for wave in range(max_waves):
        components = get_4conn_components(grid)
        singles = [(col, list(cells)[0]) for col, cells in components if len(cells) == 1]
        multis = [(col, cells) for col, cells in components if len(cells) > 1]

        new_cells = {}

        # Phase 1: Standard reflections (4-connected multis)
        for s_col, s_rc in singles:
            for m_col, m_cells in multis:
                if s_col == m_col:
                    continue
                do_reflection(s_col, s_rc, m_col, m_cells, new_cells)

        # Phase 2: 8-connected virtual shapes (only first wave)
        # Phase 3: Isolated pairs (only first wave)
        if wave == 0:
            matched_singles = set()
            for s_col, s_rc in singles:
                for m_col, m_cells in multis:
                    if s_col == m_col:
                        continue
                    sr, sc = s_rc
                    for ddr in [-1, 0, 1]:
                        for ddc in [-1, 0, 1]:
                            if ddr == 0 and ddc == 0:
                                continue
                            if (sr + ddr, sc + ddc) in m_cells:
                                matched_singles.add(s_rc)

            color_singles = {}
            for s_col, s_rc in singles:
                if s_rc not in matched_singles:
                    color_singles.setdefault(s_col, []).append(s_rc)

            # Find 8-connected groups of same-color unmatched singles
            for col, cells_list in color_singles.items():
                if len(cells_list) < 2:
                    continue
                groups = get_8conn_components(set(cells_list))
                for grp in groups:
                    if len(grp) < 2:
                        continue
                    virtual_shape = grp
                    for s_col2, s_rc2 in singles:
                        if s_col2 == col:
                            continue
                        if s_rc2 in virtual_shape:
                            continue
                        sr2, sc2 = s_rc2
                        adj = False
                        for ddr in [-1, 0, 1]:
                            for ddc in [-1, 0, 1]:
                                if ddr == 0 and ddc == 0:
                                    continue
                                if (sr2 + ddr, sc2 + ddc) in virtual_shape:
                                    adj = True
                                    break
                            if adj:
                                break
                        if adj:
                            do_reflection(s_col2, s_rc2, col, virtual_shape, new_cells)

            # Phase 3: Isolated pairs
            for s_col, s_rc in singles:
                if s_rc in matched_singles:
                    continue
                sr, sc = s_rc
                adj_singles = []
                for s_col2, s_rc2 in singles:
                    if s_col2 == s_col:
                        continue
                    r2, c2 = s_rc2
                    if abs(sr - r2) <= 1 and abs(sc - c2) <= 1 and (sr != r2 or sc != c2):
                        adj_singles.append((s_col2, s_rc2))
                if len(adj_singles) == 1 and s_rc not in matched_singles:
                    other_col, other_rc = adj_singles[0]
                    or2, oc2 = other_rc
                    other_adj = False
                    for s_col3, s_rc3 in singles:
                        if s_rc3 == s_rc or s_rc3 == other_rc:
                            continue
                        r3, c3 = s_rc3
                        if abs(or2 - r3) <= 1 and abs(oc2 - c3) <= 1:
                            other_adj = True
                            break
                    for m_col, m_cells in multis:
                        for ddr in [-1, 0, 1]:
                            for ddc in [-1, 0, 1]:
                                if ddr == 0 and ddc == 0:
                                    continue
                                if (or2 + ddr, oc2 + ddc) in m_cells:
                                    other_adj = True
                    if not other_adj:
                        dr = sr + sr - or2
                        dc = sc + sc - oc2
                        if not is_4adj(dr, dc, {other_rc}):
                            for offset in [-1, 0, 1]:
                                nr, nc = dr, dc + offset
                                if in_bounds(nr, nc) and grid[nr][nc] == bg and (nr, nc) not in new_cells:
                                    new_cells[(nr, nc)] = s_col

        if not new_cells:
            break

        for (r, c), col in new_cells.items():
            grid[r][c] = col

    return grid
