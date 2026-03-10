"""
Solver for ARC-AGI puzzle 21897d95.

Transformation rule:
1. Input is divided into colored regions (rectangular blocks or diagonal regions).
2. Arrows are T-shaped markers: 3 cells of color 1 + 1 center cell.
   If center is also color 1: no payload. Otherwise: payload = center color.
3. The center with 3 marker-1 neighbors -> the missing 4th direction's OPPOSITE
   is the arrow direction.
4. Arrow in source region -> targets adjacent region in arrow direction.
   Flow color = payload color (if present) or source region color.
5. Square grids: per-pixel color remapping (handles diagonal boundaries).
6. Non-square grids: output = post-arrow block grid rotated 90 degrees.
"""

import json
from collections import Counter, deque


def _detect_t_arrows(grid, rows, cols):
    """Detect T-shaped arrows including payload arrows where center is non-1."""
    marker1 = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                marker1.add((r, c))

    candidates = set()
    for r, c in marker1:
        candidates.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                candidates.add((nr, nc))

    arrows = []
    used = set()
    for cr, cc in sorted(candidates):
        up = (cr - 1, cc) in marker1
        dn = (cr + 1, cc) in marker1
        lt = (cr, cc - 1) in marker1
        rt = (cr, cc + 1) in marker1
        n = sum([up, dn, lt, rt])
        if n != 3:
            continue

        cells = {(cr, cc)}
        if up: cells.add((cr - 1, cc))
        if dn: cells.add((cr + 1, cc))
        if lt: cells.add((cr, cc - 1))
        if rt: cells.add((cr, cc + 1))

        if cells & used:
            continue

        if not up:
            direction = 'DOWN'
        elif not dn:
            direction = 'UP'
        elif not lt:
            direction = 'RIGHT'
        else:
            direction = 'LEFT'

        is_m = (cr, cc) in marker1
        payload = None if is_m else grid[cr][cc]
        arrows.append({
            'center': (cr, cc),
            'direction': direction,
            'payload': payload,
            'cells': cells,
        })
        used |= cells

    return arrows


def _clean_grid(grid, rows, cols, marker_cells, block_colors):
    """Replace marker cells with neighboring block colors."""
    marker_cells = set(marker_cells)
    clean = [row[:] for row in grid]
    for _ in range(20):
        changed = False
        for r, c in list(marker_cells):
            nbrs = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in marker_cells:
                    nbrs.append(clean[nr][nc])
            if nbrs:
                best = Counter(nbrs).most_common(1)[0][0]
                if best in block_colors:
                    clean[r][c] = best
                    marker_cells.discard((r, c))
                    changed = True
        if not changed:
            break
    for r, c in list(marker_cells):
        nbrs = []
        for dr2 in range(-10, 11):
            for dc2 in range(-10, 11):
                nr, nc = r + dr2, c + dc2
                if (0 <= nr < rows and 0 <= nc < cols
                        and clean[nr][nc] in block_colors
                        and (nr, nc) not in marker_cells):
                    nbrs.append(clean[nr][nc])
        if nbrs:
            clean[r][c] = Counter(nbrs).most_common(1)[0][0]
        marker_cells.discard((r, c))
    return clean


def _solve_square(grid, rows, cols, comps_by_color):
    """Per-pixel approach for square grids — handles diagonal boundaries."""
    # Block colors: every color except marker color 1
    block_colors = {c for c in comps_by_color if c != 1}

    # Detect arrows (including payload arrows)
    arrows = _detect_t_arrows(grid, rows, cols)

    # Marker cells = non-block-color cells + all arrow cells (incl. payload centers)
    marker_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in block_colors:
                marker_cells.add((r, c))
    for a in arrows:
        marker_cells |= a['cells']

    clean = _clean_grid(grid, rows, cols, marker_cells, block_colors)

    # Connected components in cleaned grid
    comp_id = {}
    comp_colors = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in comp_id:
                continue
            color = clean[r][c]
            comp = []
            q = deque([(r, c)])
            cid = len(comp_colors)
            comp_id[(r, c)] = cid
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and (nr, nc) not in comp_id
                            and clean[nr][nc] == color):
                        comp_id[(nr, nc)] = cid
                        q.append((nr, nc))
            comp_colors.append(color)

    # Map arrows to source/target components
    dir_delta = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    color_map = {}

    for arrow in arrows:
        acr, acc = arrow['center']
        direction = arrow['direction']
        payload = arrow['payload']
        dr, dc = dir_delta[direction]

        src_comp = comp_id[(acr, acc)]
        src_color = comp_colors[src_comp]

        # Walk in arrow direction to find target component
        r2, c2 = acr + dr, acc + dc
        tgt_comp = None
        while 0 <= r2 < rows and 0 <= c2 < cols:
            if comp_id[(r2, c2)] != src_comp:
                tgt_comp = comp_id[(r2, c2)]
                break
            r2 += dr
            c2 += dc

        if tgt_comp is not None:
            flow_color = payload if payload is not None else src_color
            color_map[tgt_comp] = flow_color

    return [
        [color_map.get(comp_id[(r, c)], clean[r][c]) for c in range(cols)]
        for r in range(rows)
    ]


def solve(grid):
    rows, cols = len(grid), len(grid[0])

    # --- Step 1: Identify block colors via component analysis ---
    visited = {}
    comps_by_color = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue
            color = grid[r][c]
            comp = []
            q = deque([(r, c)])
            visited[(r, c)] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == color:
                        visited[(nr, nc)] = True
                        q.append((nr, nc))
            comps_by_color.setdefault(color, []).append(comp)

    # --- Block analysis (always run) ---
    threshold = max(min(rows, cols) // 2, 5)
    block_colors = set()
    for color, comp_list in comps_by_color.items():
        if max(len(c) for c in comp_list) >= threshold:
            block_colors.add(color)

    # --- Step 2: Clean grid (replace markers with neighboring block colors) ---
    marker_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in block_colors:
                marker_cells.add((r, c))
    orig_markers = set(marker_cells)

    clean = _clean_grid(grid, rows, cols, marker_cells, block_colors)

    # --- Step 3: Find block boundaries ---
    col_sigs = [tuple(clean[r][c] for r in range(rows)) for c in range(cols)]
    col_bounds = [0]
    for c in range(1, cols):
        if col_sigs[c] != col_sigs[c - 1]:
            col_bounds.append(c)
    col_bounds.append(cols)

    # Merge thin columns (width < 2)
    ch = True
    while ch:
        ch = False
        widths = [col_bounds[i + 1] - col_bounds[i] for i in range(len(col_bounds) - 1)]
        for i, w in enumerate(widths):
            if w < 2:
                if i == 0:
                    del col_bounds[1]
                elif i == len(widths) - 1:
                    del col_bounds[-2]
                else:
                    if widths[i + 1] < 2:
                        del col_bounds[i + 1]
                    else:
                        del col_bounds[i]
                ch = True
                break
    ncb = len(col_bounds) - 1

    def row_block_sig(r):
        sig = []
        for bj in range(ncb):
            c1, c2 = col_bounds[bj], col_bounds[bj + 1]
            vals = [clean[r][c] for c in range(c1, c2)]
            sig.append(Counter(vals).most_common(1)[0][0])
        return tuple(sig)

    row_sigs = [row_block_sig(r) for r in range(rows)]
    row_bounds = [0]
    for r in range(1, rows):
        if row_sigs[r] != row_sigs[r - 1]:
            row_bounds.append(r)
    row_bounds.append(rows)

    # Smart merge: merge thin rows with their neighbor if signatures are similar
    ch = True
    while ch:
        ch = False
        nb = len(row_bounds) - 1
        widths = [row_bounds[i + 1] - row_bounds[i] for i in range(nb)]
        for i in range(nb - 1):
            if widths[i] > 1 and widths[i + 1] > 1:
                continue
            sig1 = list(row_block_sig(row_bounds[i]))
            sig2 = list(row_block_sig(row_bounds[i + 1]))
            matching = sum(1 for a, b in zip(sig1, sig2) if a == b)
            if matching >= (ncb + 1) // 2:
                del row_bounds[i + 1]
                ch = True
                break

    nrb = len(row_bounds) - 1

    def block_color(r1, r2, c1, c2):
        row_colors = []
        for r in range(r1, r2):
            vals = [clean[r][c] for c in range(c1, c2)]
            row_colors.append(Counter(vals).most_common(1)[0][0])
        return Counter(row_colors).most_common(1)[0][0]

    bgrid = []
    for bi in range(nrb):
        row = []
        for bj in range(ncb):
            row.append(block_color(row_bounds[bi], row_bounds[bi + 1],
                                   col_bounds[bj], col_bounds[bj + 1]))
        bgrid.append(row)

    # Merge identical adjacent rows/columns
    i = 0
    while i < len(bgrid) - 1:
        if bgrid[i] == bgrid[i + 1]:
            bgrid.pop(i + 1)
            del row_bounds[i + 1]
        else:
            i += 1
    j = 0
    while j < len(bgrid[0]) - 1:
        if all(bgrid[i][j] == bgrid[i][j + 1] for i in range(len(bgrid))):
            for row in bgrid:
                row.pop(j + 1)
            del col_bounds[j + 1]
        else:
            j += 1

    nrb = len(bgrid)
    ncb = len(bgrid[0])
    row_heights = [row_bounds[i + 1] - row_bounds[i] for i in range(nrb)]
    col_widths = [col_bounds[i + 1] - col_bounds[i] for i in range(ncb)]

    has_uniform_row = any(len(set(row)) == 1 for row in bgrid)

    # For square grids: check if rectangular blocks fit the cleaned grid well.
    # If not (diagonal boundaries), fall back to per-pixel approach.
    if rows == cols:
        match_count = 0
        for bi in range(nrb):
            for bj in range(ncb):
                bc = bgrid[bi][bj]
                for r in range(row_bounds[bi], row_bounds[bi + 1]):
                    for c in range(col_bounds[bj], col_bounds[bj + 1]):
                        if clean[r][c] == bc:
                            match_count += 1
        if match_count / (rows * cols) < 0.9:
            return _solve_square(grid, rows, cols, comps_by_color)

    # --- Step 4: Block-grid connected components ---
    bvisited = {}
    bcomponents = []
    for bi in range(nrb):
        for bj in range(ncb):
            if (bi, bj) in bvisited:
                continue
            color = bgrid[bi][bj]
            comp = []
            q = deque([(bi, bj)])
            bvisited[(bi, bj)] = len(bcomponents)
            while q:
                ci, cj = q.popleft()
                comp.append((ci, cj))
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nrb and 0 <= nj < ncb and (ni, nj) not in bvisited and bgrid[ni][nj] == color:
                        bvisited[(ni, nj)] = len(bcomponents)
                        q.append((ni, nj))
            bcomponents.append((color, comp))

    # --- Step 5: Find arrows ---
    mvisited = set()
    arrows = []
    for r, c in sorted(orig_markers):
        if (r, c) in mvisited:
            continue
        comp = []
        q = deque([(r, c)])
        mvisited.add((r, c))
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc, grid[cr][cc]))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in orig_markers and (nr, nc) not in mvisited:
                    mvisited.add((nr, nc))
                    q.append((nr, nc))
        if len(comp) == 4:
            arrows.append(comp)

    # --- Step 6: Process arrows ---
    color_map = {}
    for arrow in arrows:
        all_cells = set((r, c) for r, c, v in arrow)
        payload_colors = [v for r, c, v in arrow if v != 1]
        payload = payload_colors[0] if payload_colors else None

        direction = None
        center_cell = None
        for cr, cc in all_cells:
            up = (cr - 1, cc) in all_cells
            down = (cr + 1, cc) in all_cells
            left = (cr, cc - 1) in all_cells
            right = (cr, cc + 1) in all_cells
            n = sum([up, down, left, right])
            if n == 3:
                center_cell = (cr, cc)
                if not up:
                    direction = 'DOWN'
                elif not down:
                    direction = 'UP'
                elif not left:
                    direction = 'RIGHT'
                elif not right:
                    direction = 'LEFT'
                break
        if not direction:
            continue

        tcr, tcc = center_cell
        src_bi = next((bi for bi in range(nrb) if row_bounds[bi] <= tcr < row_bounds[bi + 1]), None)
        src_bj = next((bj for bj in range(ncb) if col_bounds[bj] <= tcc < col_bounds[bj + 1]), None)
        if src_bi is None or src_bj is None:
            continue

        tgt_bi, tgt_bj = src_bi, src_bj
        if direction == 'UP':
            tgt_bi -= 1
        elif direction == 'DOWN':
            tgt_bi += 1
        elif direction == 'LEFT':
            tgt_bj -= 1
        elif direction == 'RIGHT':
            tgt_bj += 1

        if not (0 <= tgt_bi < nrb and 0 <= tgt_bj < ncb):
            continue

        flow_color = payload if payload is not None else bgrid[src_bi][src_bj]
        tgt_comp_id = bvisited[(tgt_bi, tgt_bj)]
        color_map[tgt_comp_id] = flow_color

    new_bgrid = [
        [color_map.get(bvisited[(bi, bj)], bgrid[bi][bj]) for bj in range(ncb)]
        for bi in range(nrb)
    ]

    # --- Step 7: Expand and optionally rotate ---
    def expand(bg, rh, cw):
        out = []
        for bi, h in enumerate(rh):
            for _ in range(h):
                row = []
                for bj, w in enumerate(cw):
                    row.extend([bg[bi][bj]] * w)
                out.append(row)
        return out

    if nrb == ncb:
        if rows != cols:
            # Non-square pixel, square block -> CCW
            fbg = [[new_bgrid[j][ncb - 1 - i] for j in range(nrb)] for i in range(ncb)]
            frh = [col_widths[ncb - 1 - i] for i in range(ncb)]
            fcw = list(row_heights)
            return expand(fbg, frh, fcw)
        else:
            return expand(new_bgrid, row_heights, col_widths)

    # Non-square block grid
    if rows != cols:
        if has_uniform_row:
            # CCW rotation
            fbg = [[new_bgrid[j][ncb - 1 - i] for j in range(nrb)] for i in range(ncb)]
            frh = [col_widths[ncb - 1 - i] for i in range(ncb)]
            fcw = list(row_heights)
        else:
            # CW rotation
            fbg = [[new_bgrid[nrb - 1 - j][i] for j in range(nrb)] for i in range(ncb)]
            frh = list(col_widths)
            fcw = [row_heights[nrb - 1 - j] for j in range(nrb)]
        return expand(fbg, frh, fcw)
    else:
        # Square pixel grid with non-square block grid
        if has_uniform_row:
            fbg = [[new_bgrid[j][ncb - 1 - i] for j in range(nrb)] for i in range(ncb)]
            frh = [col_widths[ncb - 1 - i] for i in range(ncb)]
            fcw = list(row_heights)
            return expand(fbg, frh, fcw)
        else:
            return expand(new_bgrid, row_heights, col_widths)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/21897d95.json'
    with open(path) as f:
        task = json.load(f)

    all_pass = True
    print("=== Training ===")
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        expected = ex['output']
        ok = result == expected
        print(f"Train {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
            mr = min(len(result), len(expected))
            mc = min(len(result[0]) if result else 0, len(expected[0]) if expected else 0)
            m = sum(1 for r in range(mr) for c in range(mc) if result[r][c] != expected[r][c])
            print(f"  dims: {len(result)}x{len(result[0]) if result else 0} vs {len(expected)}x{len(expected[0])}, mismatches={m}")

    if 'test' in task and task['test']:
        print("\n=== Test ===")
        for i, ex in enumerate(task['test']):
            if 'output' not in ex:
                result = solve(ex['input'])
                print(f"Test {i}: output {len(result)}x{len(result[0]) if result else 0}")
                continue
            result = solve(ex['input'])
            expected = ex['output']
            ok = result == expected
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                mr = min(len(result), len(expected))
                mc = min(len(result[0]) if result else 0, len(expected[0]) if expected else 0)
                m = sum(1 for r in range(mr) for c in range(mc) if result[r][c] != expected[r][c])
                print(f"  dims: {len(result)}x{len(result[0]) if result else 0} vs {len(expected)}x{len(expected[0])}, mismatches={m}")

    print(f"\n{'All training examples PASS!' if all_pass else 'Some training examples FAILED.'}")
