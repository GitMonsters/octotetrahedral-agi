def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """Reflect arm pattern into 4-fold symmetry around anchor diamond center.
    
    Two non-bg colors: smaller is diamond anchor, larger is arm.
    One non-bg color: anchor is invisible (bg-colored), find center by search.
    """
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            if v != bg:
                non_bg.setdefault(v, []).append((r, c))

    output = copy.deepcopy(input_grid)

    def is_diamond(cells):
        if len(cells) != 5:
            return None
        s = set(cells)
        cr = sum(r for r, _ in cells) // 5
        cc = sum(c for _, c in cells) // 5
        expected = {(cr-1,cc-1),(cr-1,cc+1),(cr,cc),(cr+1,cc-1),(cr+1,cc+1)}
        return (cr, cc) if s == expected else None

    def arm_in_one_quadrant(arm, cr, cc):
        sr_set, sc_set = set(), set()
        for r, c in arm:
            dr, dc = r - cr, c - cc
            if dr == 0 or dc == 0:
                return False
            if abs(dr) <= 1 and abs(dc) <= 1:
                return False
            sr_set.add(1 if dr > 0 else -1)
            sc_set.add(1 if dc > 0 else -1)
        return len(sr_set) == 1 and len(sc_set) == 1

    center = None
    arm_cells = []
    arm_color = None

    if len(non_bg) >= 2:
        # Two colors: one is diamond anchor, other is arm
        colors = list(non_bg.keys())
        for i in range(len(colors)):
            for j in range(len(colors)):
                if i == j:
                    continue
                dc = is_diamond(non_bg[colors[i]])
                if dc is not None:
                    center = dc
                    arm_color = colors[j]
                    arm_cells = non_bg[colors[j]]
                    break
            if center:
                break
        if center is None:
            # Fallback: smaller group as anchor
            colors.sort(key=lambda c: len(non_bg[c]))
            ac, arm_color = colors[0], colors[1]
            cells = non_bg[ac]
            cr = sum(r for r, _ in cells) // len(cells)
            cc = sum(c for _, c in cells) // len(cells)
            center = (cr, cc)
            arm_cells = non_bg[arm_color]

    elif len(non_bg) == 1:
        arm_color = list(non_bg.keys())[0]
        all_cells = non_bg[arm_color]
        cell_set = set(all_cells)

        # Check if subset forms diamond (same color anchor)
        diamond_found = False
        for r, c in all_cells:
            expected = {(r-1,c-1),(r-1,c+1),(r,c),(r+1,c-1),(r+1,c+1)}
            if expected.issubset(cell_set):
                remaining = [p for p in all_cells if p not in expected]
                if remaining and arm_in_one_quadrant(remaining, r, c):
                    center = (r, c)
                    arm_cells = remaining
                    diamond_found = True
                    break

        if not diamond_found:
            # Invisible anchor: brute-force search for center
            arm_cells = all_cells
            best_center = None
            best_tight = float('inf')

            for cr in range(rows):
                for cc in range(cols):
                    if not arm_in_one_quadrant(arm_cells, cr, cc):
                        continue
                    # Check all reflections fit in grid
                    ok = True
                    for r, c in arm_cells:
                        dr, dc = r - cr, c - cc
                        for sr, sc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                            nr, nc = cr + sr*dr, cc + sc*dc
                            if not (0 <= nr < rows and 0 <= nc < cols):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    min_dr = min(abs(r - cr) for r, _ in arm_cells)
                    min_dc = min(abs(c - cc) for _, c in arm_cells)
                    tight = min_dr + min_dc
                    if tight < best_tight:
                        best_tight = tight
                        best_center = (cr, cc)

            center = best_center

    # Apply 4-fold reflection
    if center and arm_cells and arm_color is not None:
        cr, cc = center
        for r, c in arm_cells:
            dr, dc = r - cr, c - cc
            for sr, sc in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = cr + sr*dr, cc + sc*dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    output[nr][nc] = arm_color

    return output
