from collections import Counter, deque


def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
    result = [row[:] for row in grid]

    def ff4(sr, sc, visited):
        group = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] == bg:
                continue
            visited[r][c] = True
            group.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))
        return group

    def color_ccs(group):
        gs = set(group)
        vis = set()
        ccs = []
        for rc in group:
            if rc in vis:
                continue
            col = grid[rc[0]][rc[1]]
            cc = []
            stack = [rc]
            while stack:
                r, c = stack.pop()
                if (r, c) in vis or (r, c) not in gs or grid[r][c] != col:
                    continue
                vis.add((r, c))
                cc.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))
            ccs.append((col, frozenset(cc)))
        return ccs

    def adj_dir(cells_a, cell_b):
        rb, cb = cell_b
        for ra, ca in cells_a:
            if ra == rb and ca + 1 == cb:
                return 'RIGHT'
            if ra == rb and ca - 1 == cb:
                return 'LEFT'
            if ra + 1 == rb and ca == cb:
                return 'DOWN'
            if ra - 1 == rb and ca == cb:
                return 'UP'
        return None

    def reflect_cells(cells, direction):
        cells = list(cells)
        if direction == 'RIGHT':
            rb = max(c for r, c in cells)
            return frozenset((r, 2 * rb + 1 - c) for r, c in cells)
        if direction == 'LEFT':
            lb = min(c for r, c in cells)
            return frozenset((r, 2 * lb - 1 - c) for r, c in cells)
        if direction == 'DOWN':
            bb = max(r for r, c in cells)
            return frozenset((2 * bb + 1 - r, c) for r, c in cells)
        if direction == 'UP':
            tb = min(r for r, c in cells)
            return frozenset((2 * tb - 1 - r, c) for r, c in cells)

    def write_cells(cells, color):
        for r, c in cells:
            if 0 <= r < rows and 0 <= c < cols:
                result[r][c] = color

    # Step 1: main algorithm (4-connectivity groups, BFS chain reflections)
    computed_copies = []  # (cells, color, indicator_cell)
    visited = [[False] * cols for _ in range(rows)]
    groups = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                groups.append(ff4(r, c, visited))

    for group in groups:
        ccs = color_ccs(group)
        if len(ccs) <= 1:
            continue
        degrees = [0] * len(ccs)
        for i in range(len(ccs)):
            for j in range(len(ccs)):
                if i == j:
                    continue
                if adj_dir(ccs[i][1], next(iter(ccs[j][1]))):
                    degrees[i] += 1
        body_idx = max(range(len(ccs)), key=lambda i: (len(ccs[i][1]), degrees[i]))
        processed = {body_idx}
        queue = deque([(body_idx, ccs[body_idx][1])])
        while queue:
            cur_idx, cur_cells = queue.popleft()
            for j, (jcol, jcells) in enumerate(ccs):
                if j in processed:
                    continue
                jcell = next(iter(jcells))
                d = adj_dir(cur_cells, jcell)
                if not d:
                    continue
                processed.add(j)
                new_cells = reflect_cells(cur_cells, d)
                write_cells(new_cells, jcol)
                computed_copies.append((new_cells, jcol, jcell))
                queue.append((j, new_cells))

    # Step 2: isolated cell handling
    isolated_input_set = set(g[0] for g in groups if len(g) == 1)

    for group in groups:
        if len(group) != 1:
            continue
        r0, c0 = group[0]
        v0 = grid[r0][c0]

        # 8-adj non-bg cells that are NOT other isolated input cells
        adj8_main = [
            (r0 + dr, c0 + dc, dr, dc)
            for dr, dc in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
            if 0 <= r0 + dr < rows and 0 <= c0 + dc < cols
            and result[r0 + dr][c0 + dc] != bg
            and (r0 + dr, c0 + dc) not in isolated_input_set
        ]

        if adj8_main:
            # Diagonal chain extension: extend the contiguous same-color diagonal chain
            ar, ac, adr, adc = adj8_main[0]
            raw_dir = (adr, adc)
            if raw_dir[0] < 0 or (raw_dir[0] == 0 and raw_dir[1] < 0):
                fwd_dir = (-raw_dir[0], -raw_dir[1])
            else:
                fwd_dir = raw_dir
            dr, dc = fwd_dir

            chain_color = result[r0][c0]
            chain = [(r0, c0)]
            r, c = r0 + dr, c0 + dc
            while 0 <= r < rows and 0 <= c < cols and result[r][c] == chain_color:
                chain.append((r, c))
                r += dr
                c += dc
            r, c = r0 - dr, c0 - dc
            while 0 <= r < rows and 0 <= c < cols and result[r][c] == chain_color:
                chain.insert(0, (r, c))
                r -= dr
                c -= dc

            chain_set = set(chain)
            chain.sort(key=lambda rc: rc[0])

            # Extend chain by 1 step at the tail
            tail_r, tail_c = chain[-1]
            ext_r, ext_c = tail_r + dr, tail_c + dc
            if 0 <= ext_r < rows and 0 <= ext_c < cols and result[ext_r][ext_c] == bg:
                result[ext_r][ext_c] = chain_color

            # Find lateral cells: 4-adj to chain, not in chain, non-bg
            lateral = {}
            for cr, cc in chain:
                for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    lr, lc = cr + ddr, cc + ddc
                    if 0 <= lr < rows and 0 <= lc < cols:
                        if result[lr][lc] != bg and (lr, lc) not in chain_set:
                            lateral[(lr, lc)] = result[lr][lc]

            # Extend each lateral 90 degrees clockwise of the chain direction
            ext_dr, ext_dc = dc, -dr
            for (lr, lc), lv in lateral.items():
                nlr, nlc = lr + ext_dr, lc + ext_dc
                if 0 <= nlr < rows and 0 <= nlc < cols and result[nlr][nlc] == bg:
                    result[nlr][nlc] = lv

        else:
            # Secondary copy rule: find computed copy in the same row range to the right
            best_copy = None
            best_dist = float('inf')
            for (copy_cells, copy_color, ind_cell) in computed_copies:
                copy_rows = set(r for r, c in copy_cells)
                if r0 not in copy_rows:
                    continue
                max_copy_col = max(c for r, c in copy_cells)
                if c0 <= max_copy_col:
                    continue
                dist = c0 - max_copy_col
                if dist < best_dist:
                    best_dist = dist
                    best_copy = (copy_cells, copy_color, ind_cell)

            if best_copy is None:
                continue

            copy_cells, copy_color, ind_cell = best_copy
            ind_c = ind_cell[1]
            all_copy_cols = sorted(set(c for r, c in copy_cells))
            W = len(all_copy_cols)

            # Keep inner 3 columns closest to the indicator; drop the outermost
            if W <= 3:
                inner_cols = set(all_copy_cols)
            elif ind_c == all_copy_cols[0]:
                inner_cols = set(all_copy_cols[:3])   # indicator at left -> keep left 3
            else:
                inner_cols = set(all_copy_cols[-3:])  # indicator at right -> keep right 3

            inner_cells = frozenset((r, c) for r, c in copy_cells if c in inner_cols)
            inner_sorted = sorted(inner_cols)
            opp_edge = max(inner_sorted) if ind_c == min(inner_sorted) else min(inner_sorted)
            shift = c0 - opp_edge
            write_cells(frozenset((r, c + shift) for r, c in inner_cells), v0)

    return result
