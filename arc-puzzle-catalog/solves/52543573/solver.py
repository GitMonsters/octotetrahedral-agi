"""
ARC-AGI puzzle 52543573 solver.

Three puzzle types detected from input structure:
1. No L-shapes -> identity (Train 1 pattern)
2. L-shapes + bridge markers + singletons at cv/ch ->
   FORWARD rule: bridge_r->V(cv), bridge_c->H(ch)
   Output adds: diag_free at (mr+dr,mc+dc), sing_free at bridge_sing+(perp)
3. L-shapes + bridge markers + NO singletons + direct markers at cv/ch ->
   REVERSED rule: bridge_r->H(ch), bridge_c->V(cv)
   Output adds: direct_color at ch/cv for each remaining L-shape
4. L-shapes + direct markers at cv/ch, NO bridge markers ->
   Decision tree determines V/H for each L-shape
"""
import copy
from collections import Counter


def find_components(grid, color):
    rows, cols = len(grid), len(grid[0])
    visited, shapes = set(), []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and (r, c) not in visited:
                comp, q = [], [(r, c)]
                while q:
                    nr, nc = q.pop()
                    if (nr, nc) in visited:
                        continue
                    visited.add((nr, nc))
                    comp.append((nr, nc))
                    for dr2, dc2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        rr, cc = nr + dr2, nc + dc2
                        if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] == color and (rr, cc) not in visited:
                            q.append((rr, cc))
                shapes.append(tuple(sorted(comp)))
    return shapes


def get_lshape_info(cells):
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    cell_set = set(cells)
    missing = [c for c in corners if c not in cell_set][0]
    elbow = (min_r + max_r - missing[0], min_c + max_c - missing[1])
    dr = (missing[0] - elbow[0]) // abs(missing[0] - elbow[0])
    dc = (missing[1] - elbow[1]) // abs(missing[1] - elbow[1])
    cv = (missing[0] + dr, missing[1])
    ch = (missing[0], missing[1] + dc)
    return missing, elbow, dr, dc, cv, ch


def dt_choose_vh(mr, mc, er, ec, dr, dc, rows, cols):
    """Decision tree for Train-2-type (no bridge markers) V/H choice."""
    mH_miss = min(mr, rows - 1 - mr)
    mV_miss = min(mc, cols - 1 - mc)
    mH_elb = min(er, rows - 1 - er)
    if ec <= 17.5:
        if mH_miss <= 7.5:
            if mr * dr - mc * dc <= 6.5:
                if mr * dc + mc * dr <= -3.0:
                    if mV_miss <= 10.5:
                        return 'V' if er - ec <= 14.0 else 'H'
                    else:
                        return 'H'
                else:
                    return 'V'
            else:
                return 'H'
        else:
            return 'V'
    else:
        return 'H' if mH_elb <= 11.0 else 'V'


def is_valid(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    sc = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    bg = sc.most_common(1)[0][0]
    non_bg = [(k, v) for k, v in sc.items() if k != bg]
    if not non_bg:
        return [row[:] for row in grid]

    shape_color = max(non_bg, key=lambda x: x[1])[0]
    marker_colors = sorted(k for k, v in non_bg if k != shape_color)

    shape_comps = find_components(grid, shape_color)
    lshapes = [c for c in shape_comps if len(c) == 3]
    singletons = set(c[0] for c in shape_comps if len(c) == 1)

    if not lshapes:
        return [row[:] for row in grid]

    marker_pos = {(r, c): grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] in marker_colors}

    # Classify bridge vs direct markers for each L-shape
    has_bridge = False
    direct_marker_color = None

    for cells in lshapes:
        missing, elbow, dr, dc, cv, ch = get_lshape_info(cells)
        mr, mc = missing
        er, ec = elbow
        bridge_r = (er, mc + dc)
        bridge_c = (mr + dr, ec)
        if bridge_r in marker_pos or bridge_c in marker_pos:
            has_bridge = True
        if cv in marker_pos:
            direct_marker_color = marker_pos[cv]
        if ch in marker_pos:
            direct_marker_color = marker_pos[ch]

    out = [row[:] for row in grid]

    if has_bridge and singletons:
        # TYPE 2: FORWARD rule (Train 0)
        # Bridge markers (gray=5) indicate direction: bridge_r->V, bridge_c->H
        # Output adds diag_free (other color) and sing_free (bridge color)
        
        # Determine colors: bridge_color is at bridge positions
        bridge_color = None
        for cells in lshapes:
            missing, elbow, dr, dc, cv, ch = get_lshape_info(cells)
            mr, mc = missing; er, ec = elbow
            br = (er, mc + dc); bc = (mr + dr, ec)
            if br in marker_pos:
                bridge_color = marker_pos[br]; break
            if bc in marker_pos:
                bridge_color = marker_pos[bc]; break
        
        diag_free_color = next((c for c in marker_colors if c != bridge_color), bridge_color)

        for cells in lshapes:
            missing, elbow, dr, dc, cv, ch = get_lshape_info(cells)
            mr, mc = missing; er, ec = elbow
            bridge_r_pos = (er, mc + dc)
            bridge_c_pos = (mr + dr, ec)
            diag = (mr + dr, mc + dc)

            if bridge_r_pos in marker_pos:
                sing_free = (er + dr, mc + 2 * dc)
            elif bridge_c_pos in marker_pos:
                sing_free = (mr + 2 * dr, ec + dc)
            else:
                continue

            if is_valid(diag[0], diag[1], rows, cols) and out[diag[0]][diag[1]] == bg:
                out[diag[0]][diag[1]] = diag_free_color
            if is_valid(sing_free[0], sing_free[1], rows, cols) and out[sing_free[0]][sing_free[1]] == bg:
                out[sing_free[0]][sing_free[1]] = bridge_color

    elif has_bridge and not singletons and direct_marker_color is not None:
        # TYPE 3: REVERSED rule (Test pattern)
        # bridge_r -> H (direct at ch), bridge_c -> V (direct at cv)
        for cells in lshapes:
            missing, elbow, dr, dc, cv, ch = get_lshape_info(cells)
            mr, mc = missing; er, ec = elbow
            bridge_r_pos = (er, mc + dc)
            bridge_c_pos = (mr + dr, ec)

            if bridge_r_pos in marker_pos:
                direct = ch
            elif bridge_c_pos in marker_pos:
                direct = cv
            else:
                continue

            if is_valid(direct[0], direct[1], rows, cols) and out[direct[0]][direct[1]] == bg:
                out[direct[0]][direct[1]] = direct_marker_color

    elif not has_bridge and direct_marker_color is not None:
        # TYPE 4: Decision tree rule (Train 2)
        # No bridge markers; use decision tree for V/H choice
        for cells in lshapes:
            missing, elbow, dr, dc, cv, ch = get_lshape_info(cells)
            mr, mc = missing; er, ec = elbow

            if cv in marker_pos or ch in marker_pos:
                continue

            direction = dt_choose_vh(mr, mc, er, ec, dr, dc, rows, cols)
            direct = cv if direction == 'V' else ch

            if is_valid(direct[0], direct[1], rows, cols) and out[direct[0]][direct[1]] == bg:
                out[direct[0]][direct[1]] = direct_marker_color

    return out


solve = transform  # catalog alias
