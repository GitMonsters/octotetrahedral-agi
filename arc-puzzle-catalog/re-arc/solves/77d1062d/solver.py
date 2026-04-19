"""Solver for ARC puzzle 77d1062d.

Unified rule: template-based pattern propagation.
- Single non-bg color: place scaled copies of the template at computed positions
- Two colors, all dual-color components: fill bounding boxes with subordinate color
- Two colors, mixed components: stamp template pattern at seed positions
"""
from collections import Counter, deque


def transform(grid):
    R = len(grid)
    C = len(grid[0])

    colors = Counter(v for row in grid for v in row)
    bg = colors.most_common(1)[0][0]
    non_bg_colors = sorted(c for c in colors if c != bg)

    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r, c) not in visited:
                comp = []
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc, grid[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited and grid[nr][nc] != bg:
                                visited.add((nr, nc))
                                q.append((nr, nc))
                components.append(comp)

    comp_colors = [set(v for _, _, v in comp) for comp in components]
    has_dual = any(len(cc) >= 2 for cc in comp_colors)
    all_dual = all(len(cc) >= 2 for cc in comp_colors)
    has_single = any(len(cc) == 1 for cc in comp_colors)

    output = [row[:] for row in grid]

    if len(non_bg_colors) == 1:
        return _handle_single_color(grid, bg, non_bg_colors[0], components, R, C)

    if len(non_bg_colors) == 2 and all_dual:
        return _handle_bbox_fill(grid, bg, non_bg_colors, components, R, C)

    if len(non_bg_colors) >= 2 and has_dual and has_single:
        return _handle_stamp(grid, bg, non_bg_colors, components, comp_colors, R, C)

    return output


def _handle_single_color(grid, bg, color, components, R, C):
    output = [row[:] for row in grid]
    if not components:
        return output

    comp = components[0]
    cells = [(r, c) for r, c, v in comp]
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    tr, tc = min(rows), min(cols)
    th = max(rows) - tr + 1
    tw = max(cols) - tc + 1

    tmpl = [[0] * tw for _ in range(th)]
    for r, c in cells:
        tmpl[r - tr][c - tc] = 1
    tmpl_f = [row[::-1] for row in tmpl]

    def place_copy(out, template, tl_r, tl_c, scale):
        for i in range(len(template)):
            for j in range(len(template[0])):
                if template[i][j]:
                    for si in range(scale):
                        for sj in range(scale):
                            pr = tl_r + i * scale + si
                            pc = tl_c + j * scale + sj
                            if 0 <= pr < R and 0 <= pc < C:
                                out[pr][pc] = color

    copies = [
        (0, -tc, tmpl_f, 1),
        (th - 1, C - tc - th - 2, tmpl, 1),
        (R - 2 * th, -(R - 2 * th), tmpl, 1),
        (2 * th, -(tc - 1), tmpl, 2),
        (2 * th - 1, -(2 * th + 1), tmpl_f, 2),
        (R - 3 * th, th - 1, tmpl, 2),
        (R - 2 * th - 1, -2 * th, tmpl_f, 2),
    ]

    for dr, dc, t, scale in copies:
        place_copy(output, t, tr + dr, tc + dc, scale)

    return output


def _handle_bbox_fill(grid, bg, non_bg_colors, components, R, C):
    output = [row[:] for row in grid]

    # Fill color = min of non-bg colors (the subordinate/structural color)
    fill_color = min(non_bg_colors)

    for comp in components:
        rows = [r for r, c, v in comp]
        cols = [c for r, c, v in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if output[r][c] == bg:
                    output[r][c] = fill_color

    return output


def _handle_stamp(grid, bg, non_bg_colors, components, comp_colors, R, C):
    output = [row[:] for row in grid]

    templates = [i for i, cc in enumerate(comp_colors) if len(cc) >= 2]
    seeds = [i for i, cc in enumerate(comp_colors) if len(cc) == 1]

    if not templates:
        return output

    template_comp = components[templates[0]]
    template_color_set = comp_colors[templates[0]]

    seed_color_set = set()
    for i in seeds:
        seed_color_set.update(comp_colors[i])

    if not seed_color_set:
        return output

    anchor_color = list(seed_color_set)[0]
    stamp_color = [c for c in template_color_set if c != anchor_color][0]

    anchor_cells = [(r, c) for r, c, v in template_comp if v == anchor_color]
    stamp_cells = [(r, c) for r, c, v in template_comp if v == stamp_color]

    if not anchor_cells or not stamp_cells:
        return output

    anchor = anchor_cells[0]
    stamp_offsets = [(r - anchor[0], c - anchor[1]) for r, c in stamp_cells]

    for i in seeds:
        seed_comp = components[i]
        seed_cells_list = [(r, c) for r, c, v in seed_comp]

        srows = [r for r, c in seed_cells_list]
        scols = [c for r, c in seed_cells_list]
        seed_h = max(srows) - min(srows) + 1
        seed_w = max(scols) - min(scols) + 1
        scale = max(seed_h, seed_w)

        seed_center_r = (min(srows) + max(srows)) / 2.0
        seed_center_c = (min(scols) + max(scols)) / 2.0

        qr = 1 if seed_center_r > anchor[0] else -1
        qc = 1 if seed_center_c > anchor[1] else -1
        sign_r = -qr * qc
        sign_c = qr

        for sr, sc in seed_cells_list:
            for dr, dc in stamp_offsets:
                nr = sr + sign_r * dr * scale
                nc = sc + sign_c * dc * scale
                nr_int = int(round(nr))
                nc_int = int(round(nc))
                if 0 <= nr_int < R and 0 <= nc_int < C:
                    if output[nr_int][nc_int] == bg:
                        output[nr_int][nc_int] = stamp_color

    return output
