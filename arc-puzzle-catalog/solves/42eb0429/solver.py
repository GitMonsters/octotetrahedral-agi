"""Solver for 42eb0429: Template shape is tiled from scatter cells in specific directions.
Each scatter cell spawns copies of the template in its color, tiling away from the template."""

from collections import Counter, defaultdict

def transform(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cnt = Counter(c for row in grid for c in row)
    bg = cnt.most_common(1)[0][0]

    def get_all_comps():
        visited = [[False]*W for _ in range(H)]
        comps = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg and not visited[r][c]:
                    color = grid[r][c]
                    stack = [(r, c)]; visited[r][c] = True; cells = []
                    while stack:
                        cr, cc = stack.pop(); cells.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True; stack.append((nr, nc))
                    comps.append((color, cells))
        return comps

    comps = get_all_comps()
    if not comps:
        return out

    color_comps = defaultdict(list)
    for color, cells in comps:
        color_comps[color].append(cells)

    def merge_nearby_comps(comp_list, max_gap=1):
        if len(comp_list) <= 1:
            return comp_list
        bboxes = []
        for cells in comp_list:
            rmin = min(r for r, c in cells); rmax = max(r for r, c in cells)
            cmin = min(c for r, c in cells); cmax = max(c for r, c in cells)
            bboxes.append((rmin, cmin, rmax, cmax))
        parent = list(range(len(comp_list)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a
        for i in range(len(comp_list)):
            for j in range(i+1, len(comp_list)):
                r1min, c1min, r1max, c1max = bboxes[i]
                r2min, c2min, r2max, c2max = bboxes[j]
                rgap = max(0, max(r1min - r2max, r2min - r1max))
                cgap = max(0, max(c1min - c2max, c2min - c1max))
                if rgap <= max_gap and cgap <= max_gap:
                    union(i, j)
        groups = defaultdict(list)
        for i in range(len(comp_list)):
            groups[find(i)].extend(comp_list[i])
        return list(groups.values())

    best_template = None
    best_size = 0
    for color, comp_list in color_comps.items():
        merged = merge_nearby_comps(comp_list)
        for group in merged:
            if len(group) > best_size:
                best_size = len(group)
                best_template = group

    if not best_template:
        return out

    template_cells = best_template
    template_set = set(template_cells)
    trmin = min(r for r, c in template_cells)
    tcmin = min(c for r, c in template_cells)
    template_rel = [(r - trmin, c - tcmin) for r, c in template_cells]
    th = max(r for r, c in template_rel) + 1
    tw = max(c for r, c in template_rel) + 1
    spacing_r = th + 1
    spacing_c = tw + 1

    scatter_by_color = defaultdict(set)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r, c) not in template_set:
                scatter_by_color[grid[r][c]].add((r, c))

    stamp_assignments = defaultdict(lambda: defaultdict(set))
    for color, scells in scatter_by_color.items():
        for sr, sc in scells:
            for tr, tc in template_rel:
                dr = sr - trmin - tr
                dc = sc - tcmin - tc
                if spacing_r != 0 and dr % spacing_r == 0 and spacing_c != 0 and dc % spacing_c == 0:
                    n = dr // spacing_r
                    m = dc // spacing_c
                    if n == 0 and m == 0:
                        continue
                    stamp_assignments[color][(n, m)].add((sr, sc))

    for color, nm_dict in stamp_assignments.items():
        for (n, m), cells in nm_dict.items():
            org_r = trmin + n * spacing_r
            org_c = tcmin + m * spacing_c
            ok = True
            for tr, tc in template_rel:
                gr, gc = org_r + tr, org_c + tc
                if 0 <= gr < H and 0 <= gc < W:
                    v = grid[gr][gc]
                    if v != bg and v != color and (gr, gc) not in template_set:
                        ok = False
                        break
            if not ok or len(cells) == 0:
                continue

            dn = 1 if n > 0 else (-1 if n < 0 else 0)
            dm = 1 if m > 0 else (-1 if m < 0 else 0)
            cn, cm = n, m
            while True:
                if cn == 0 and cm == 0:
                    cn += dn; cm += dm
                    continue
                sr = trmin + cn * spacing_r
                sc = tcmin + cm * spacing_c
                any_visible = False
                for tr, tc in template_rel:
                    gr, gc = sr + tr, sc + tc
                    if 0 <= gr < H and 0 <= gc < W:
                        any_visible = True
                        out[gr][gc] = color
                if not any_visible:
                    break
                cn += dn; cm += dm

    return out
