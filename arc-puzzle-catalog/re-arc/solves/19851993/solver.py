def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import deque, defaultdict

    grid = [row[:] for row in input_grid]
    R, C = len(grid), len(grid[0])

    # Find background color (most common)
    color_counts: dict[int, int] = {}
    for r in range(R):
        for c in range(C):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    bg = max(color_counts, key=color_counts.get)

    # Find 3x3 template: filler at cross positions, bg at one diagonal, non-bg at other
    template_pos = None
    filler = None
    for r in range(R - 2):
        for c in range(C - 2):
            sub = [[grid[r + dr][c + dc] for dc in range(3)] for dr in range(3)]
            cross = [sub[0][1], sub[1][0], sub[1][2], sub[2][1]]
            if len(set(cross)) != 1 or cross[0] == bg:
                continue
            f = cross[0]
            # Center must be bg or same as filler
            if sub[1][1] != bg and sub[1][1] != f:
                continue
            main_ok = sub[0][0] != bg and sub[2][2] != bg
            anti_ok = sub[0][2] != bg and sub[2][0] != bg
            main_bg = sub[0][0] == bg and sub[2][2] == bg
            anti_bg = sub[0][2] == bg and sub[2][0] == bg
            if (anti_bg and main_ok) or (main_bg and anti_ok):
                template_pos = (r, c)
                filler = f
                break
        if template_pos:
            break

    tr, tc = template_pos
    template_cells = {(tr + dr, tc + dc) for dr in range(3) for dc in range(3)}

    # Find connected rectangular blocks (excluding template cells)
    visited: set[tuple[int, int]] = set()
    blocks: list[tuple[int, int, int, int]] = []  # (top_r, top_c, size, color)

    for r in range(R):
        for c in range(C):
            if (r, c) in visited or (r, c) in template_cells or grid[r][c] == bg:
                continue
            color = grid[r][c]
            q = deque([(r, c)])
            cells: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                if (cr, cc) in cells:
                    continue
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < R and 0 <= nc < C and (nr, nc) not in cells
                            and (nr, nc) not in template_cells and grid[nr][nc] == color):
                        q.append((nr, nc))
            visited |= cells
            min_r = min(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_r = max(r for r, c in cells)
            h = max_r - min_r + 1
            blocks.append((min_r, min_c, h, color))

    # Group blocks by size, then pair blocks of different colors at diagonal distance 2*s
    by_size: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for b in blocks:
        by_size[b[2]].append(b)

    for s, blks in by_size.items():
        used: set[int] = set()
        for i in range(len(blks)):
            if i in used:
                continue
            for j in range(i + 1, len(blks)):
                if j in used:
                    continue
                r1, c1, _, col1 = blks[i]
                r2, c2, _, col2 = blks[j]
                if col1 == col2:
                    continue
                dr, dc = r2 - r1, c2 - c1
                if abs(dr) == 2 * s and abs(dc) == 2 * s:
                    used.add(i)
                    used.add(j)

                    # Compute stamp origin based on direction
                    dr_s = 1 if dr > 0 else -1
                    dc_s = 1 if dc > 0 else -1
                    stamp_r = r1 - max(0, -dr_s) * 2 * s
                    stamp_c = c1 - max(0, -dc_s) * 2 * s

                    # Place filler at the 4 cross positions of scaled 3x3 template
                    for fp_r, fp_c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                        for di in range(s):
                            for dj in range(s):
                                gr = stamp_r + fp_r * s + di
                                gc = stamp_c + fp_c * s + dj
                                if 0 <= gr < R and 0 <= gc < C:
                                    grid[gr][gc] = filler
                    break

    return grid
