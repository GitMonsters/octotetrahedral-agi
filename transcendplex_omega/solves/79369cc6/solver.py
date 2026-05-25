def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Find the 4+6 template (bounding box of all 4-cells, which contains only 4s and 6s).
    Search the grid for locations where the 6-pattern matches under any of 8 rigid
    transformations (rotations + reflections). At each match, fill in 4s where the
    transformed template has 4s.
    """
    rows = len(grid)
    cols = len(grid[0])

    fours = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 4]
    if not fours:
        return [row[:] for row in grid]

    min_r = min(r for r, _ in fours)
    max_r = max(r for r, _ in fours)
    min_c = min(c for _, c in fours)
    max_c = max(c for _, c in fours)
    t_rows = max_r - min_r + 1
    t_cols = max_c - min_c + 1

    four_rel = [(r - min_r, c - min_c) for r, c in fours]
    six_rel = [
        (r - min_r, c - min_c)
        for r in range(min_r, max_r + 1)
        for c in range(min_c, max_c + 1)
        if grid[r][c] == 6
    ]

    def xform(r: int, c: int, tr: int, tc: int, t: int) -> tuple[int, int]:
        if t == 0: return (r, c)
        if t == 1: return (c, tr - 1 - r)
        if t == 2: return (tr - 1 - r, tc - 1 - c)
        if t == 3: return (tc - 1 - c, r)
        if t == 4: return (r, tc - 1 - c)
        if t == 5: return (tr - 1 - r, c)
        if t == 6: return (c, r)
        return (tc - 1 - c, tr - 1 - r)

    result = [row[:] for row in grid]
    seen: set[tuple[frozenset, frozenset]] = set()

    for t in range(8):
        tf = [xform(r, c, t_rows, t_cols, t) for r, c in four_rel]
        ts = [xform(r, c, t_rows, t_cols, t) for r, c in six_rel]
        all_p = tf + ts
        if not all_p:
            continue
        off_r = min(r for r, _ in all_p)
        off_c = min(c for _, c in all_p)
        tf = [(r - off_r, c - off_c) for r, c in tf]
        ts = [(r - off_r, c - off_c) for r, c in ts]

        key = (frozenset(tf), frozenset(ts))
        if key in seen:
            continue
        seen.add(key)

        max_dr = max(r for r, _ in all_p) - off_r
        max_dc = max(c for _, c in all_p) - off_c

        for r in range(-max_dr, rows):
            for c in range(-max_dc, cols):
                ok = True
                for dr, dc in ts:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] != 6:
                        ok = False
                        break
                if ok:
                    # Reject if any 4-position in the grid has value 6
                    for dr, dc in tf:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == 6:
                                ok = False
                                break
                if ok:
                    for dr, dc in tf:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = 4

    return result
