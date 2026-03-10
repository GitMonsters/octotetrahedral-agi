"""
ARC-AGI solver for task e12f9a14.

Each shape is a 2x2 interior surrounded by a border of 3-valued cells.
Gaps in the border emit rays of the interior color outward (cardinal gaps
go perpendicular to the wall, diagonal gaps go diagonally). When rays from
different-colored shapes converge, they redirect along the combined direction.
"""
import json
import copy


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def solve(grid):
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    out = copy.deepcopy(grid)
    shapes = find_shapes(grid, rows, cols, bg)

    rays = []
    group_dir = {}
    ray_meta = []
    shape_rays = {}
    interacted = set()

    for si, (interior, border_cells, bv, color) in enumerate(shapes):
        gaps = find_gaps(grid, rows, cols, bg, interior, border_cells)
        shape_rays[si] = []
        for gr, gc, dr, dc, is_cardinal in gaps:
            gid = len(rays)
            path = _straight_path(gr, gc, dr, dc, rows, cols)
            rays.append({
                'color': color, 'path': path,
                'group': gid
            })
            group_dir[gid] = (dr, dc)
            ray_meta.append({'shape': si, 'is_cardinal': is_cardinal, 'dir': (dr, dc)})
            shape_rays[si].append(gid)

    for _ in range(300):
        conflict = _find_conflict(rays)
        if conflict is None:
            break
        involved = _resolve(rays, group_dir, conflict, rows, cols)
        interacted.update(involved)

    # Suppress inactive diagonal gaps: a diagonal gap in a shape with exactly
    # 2 cardinal + 1 diagonal is suppressed when (a) its direction opposes both
    # cardinals, (b) it never interacted, and (c) exactly one cardinal did.
    if len(shapes) > 1:
        for si in shape_rays:
            idxs = shape_rays[si]
            card_idxs = [i for i in idxs if ray_meta[i]['is_cardinal']]
            diag_idxs = [i for i in idxs if not ray_meta[i]['is_cardinal']]
            if len(card_idxs) != 2 or len(diag_idxs) != 1:
                continue
            di = diag_idxs[0]
            dd = ray_meta[di]['dir']
            strict_opp = True
            for ci in card_idxs:
                cd = ray_meta[ci]['dir']
                for comp in (0, 1):
                    if dd[comp] != 0 and cd[comp] != 0:
                        if sign(dd[comp]) != -sign(cd[comp]):
                            strict_opp = False
            diag_int = di in interacted
            card_int = sum(1 for ci in card_idxs if ci in interacted)
            if strict_opp and not diag_int and card_int == 1:
                rays[di]['path'] = []

    for ray in rays:
        for r, c in ray['path']:
            if 0 <= r < rows and 0 <= c < cols:
                out[r][c] = ray['color']
    return out


def _straight_path(r, c, dr, dc, rows, cols):
    path = []
    while 0 <= r < rows and 0 <= c < cols:
        path.append((r, c))
        r += dr
        c += dc
    return path


def _bres_path(r, c, sum_dr, sum_dc, rows, cols):
    if sum_dr == 0 and sum_dc == 0:
        return []
    abs_dr, abs_dc = abs(sum_dr), abs(sum_dc)
    s_dr, s_dc = sign(sum_dr), sign(sum_dc)
    path = []
    err = 0
    for _ in range(rows + cols + 10):
        if abs_dr >= abs_dc:
            r += s_dr
            err += abs_dc
            if abs_dr > 0 and err >= abs_dr:
                c += s_dc
                err -= abs_dr
        else:
            c += s_dc
            err += abs_dr
            if abs_dc > 0 and err >= abs_dc:
                r += s_dr
                err -= abs_dc
        if not (0 <= r < rows and 0 <= c < cols):
            break
        path.append((r, c))
    return path


def _find_conflict(rays):
    max_len = max((len(r['path']) for r in rays), default=0)
    if max_len == 0:
        return None
    trail = [dict() for _ in rays]

    for t in range(max_len):
        heads = [
            (ri, ray['path'][t])
            for ri, ray in enumerate(rays)
            if t < len(ray['path'])
        ]

        # head-head
        for i in range(len(heads)):
            for j in range(i + 1, len(heads)):
                ri, (ra, ca) = heads[i]
                rj, (rb, cb) = heads[j]
                if rays[ri]['color'] == rays[rj]['color']:
                    continue
                if rays[ri]['group'] == rays[rj]['group']:
                    continue
                if ra == rb and ca == cb:
                    return {'type': 'head_same', 'ray_a': ri, 'ray_b': rj, 'tick': t}
                if abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                    if t + 1 < len(rays[ri]['path']) and t + 1 < len(rays[rj]['path']):
                        ra1, ca1 = rays[ri]['path'][t + 1]
                        rb1, cb1 = rays[rj]['path'][t + 1]
                        dx, dy = ca - cb, ra - rb
                        dx1, dy1 = ca1 - cb1, ra1 - rb1
                        crossed = ra1 == rb1 and ca1 == cb1
                        if dx != 0 and dx * dx1 < 0:
                            crossed = True
                        if dy != 0 and dy * dy1 < 0:
                            crossed = True
                        if crossed:
                            return {'type': 'cross', 'ray_a': ri, 'ray_b': rj, 'tick': t}

        # head-trail
        for ri, ray in enumerate(rays):
            if t >= len(ray['path']):
                continue
            hr, hc = ray['path'][t]
            for rj in range(len(rays)):
                if ri == rj:
                    continue
                if ray['color'] == rays[rj]['color']:
                    continue
                if ray['group'] == rays[rj]['group']:
                    continue
                if (hr, hc) in trail[rj] and trail[rj][(hr, hc)] < t:
                    return {
                        'type': 'head_at_trail',
                        'head_ray': ri, 'head_tick': t,
                        'trail_ray': rj, 'trail_tick': trail[rj][(hr, hc)]
                    }
                best_adj = None
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        adj = (hr + dr, hc + dc)
                        if adj in trail[rj] and trail[rj][adj] < t:
                            tt = trail[rj][adj]
                            if best_adj is None or tt < best_adj[1]:
                                best_adj = (adj, tt)
                if best_adj:
                    return {
                        'type': 'head_adj_trail',
                        'head_ray': ri, 'head_tick': t,
                        'trail_ray': rj, 'trail_tick': best_adj[1],
                        'trail_cell': best_adj[0]
                    }

        for ri, ray in enumerate(rays):
            if t < len(ray['path']):
                r, c = ray['path'][t]
                if (r, c) not in trail[ri]:
                    trail[ri][(r, c)] = t
    return None


def _resolve(rays, group_dir, conflict, rows, cols):
    ctype = conflict['type']
    if ctype == 'head_same':
        ri_a, ri_b = conflict['ray_a'], conflict['ray_b']
        trunc_a = trunc_b = conflict['tick'] - 1
    elif ctype == 'cross':
        ri_a, ri_b = conflict['ray_a'], conflict['ray_b']
        trunc_a = trunc_b = conflict['tick']
    elif ctype == 'head_at_trail':
        ri_a, ri_b = conflict['head_ray'], conflict['trail_ray']
        trunc_a = conflict['head_tick'] - 1
        trunc_b = conflict['trail_tick'] - 1
    elif ctype == 'head_adj_trail':
        ri_a, ri_b = conflict['head_ray'], conflict['trail_ray']
        trunc_a = conflict['head_tick']
        trunc_b = conflict['trail_tick']
    else:
        return set()

    ga, gb = rays[ri_a]['group'], rays[ri_b]['group']
    da, db = group_dir[ga], group_dir[gb]
    new_dr = sign(da[0]) + sign(db[0])
    new_dc = sign(da[1]) + sign(db[1])
    new_gid = min(ga, gb)
    group_dir[new_gid] = (new_dr, new_dc)

    involved = set()
    for k in range(len(rays)):
        if rays[k]['group'] == ga:
            ti = max(0, trunc_a)
            p = rays[k]['path'][:ti + 1]
            if p:
                p.extend(_bres_path(p[-1][0], p[-1][1], new_dr, new_dc, rows, cols))
            rays[k]['path'] = p
            rays[k]['group'] = new_gid
            involved.add(k)
        elif rays[k]['group'] == gb:
            ti = max(0, trunc_b)
            p = rays[k]['path'][:ti + 1]
            if p:
                p.extend(_bres_path(p[-1][0], p[-1][1], new_dr, new_dc, rows, cols))
            rays[k]['path'] = p
            rays[k]['group'] = new_gid
            involved.add(k)
    return involved


def find_shapes(grid, rows, cols, bg):
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]
    visited = set()
    shapes = []
    for pos in sorted(non_bg):
        if pos in visited:
            continue
        comp = []
        queue = [pos]
        visited.add(pos)
        while queue:
            cr, cc = queue.pop(0)
            comp.append((cr, cc, grid[cr][cc]))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in non_bg and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        vals = {}
        for r, c, v in comp:
            vals[v] = vals.get(v, 0) + 1
        if len(vals) < 2:
            continue
        sv = sorted(vals.items(), key=lambda x: x[1])
        ic, bv = sv[0][0], sv[1][0]
        interior = set((r, c) for r, c, v in comp if v == ic)
        border = set((r, c) for r, c, v in comp if v == bv)
        shapes.append((interior, border, bv, ic))
    return shapes


def find_gaps(grid, rows, cols, bg, interior, border_cells):
    expected = set()
    for r, c in interior:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) not in interior and 0 <= nr < rows and 0 <= nc < cols:
                    expected.add((nr, nc))
    gaps = []
    for gr, gc in sorted(expected - border_cells):
        if grid[gr][gc] != bg:
            continue
        card = [
            (dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (gr + dr, gc + dc) in interior
        ]
        diag = [
            (dr, dc) for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            if (gr + dr, gc + dc) in interior
        ]
        if card:
            d = card[0]
            gaps.append((gr, gc, -d[0], -d[1], True))
        elif diag:
            d = diag[0]
            gaps.append((gr, gc, -d[0], -d[1], False))
    return gaps


if __name__ == '__main__':
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/e12f9a14.json') as f:
        data = json.load(f)

    all_pass = True
    for i, pair in enumerate(data['train']):
        result = solve(pair['input'])
        expected = pair['output']
        if result == expected:
            print(f"Train {i}: PASS")
        else:
            all_pass = False
            diffs = sum(
                1 for r in range(len(expected))
                for c in range(len(expected[0]))
                if result[r][c] != expected[r][c]
            )
            print(f"Train {i}: FAIL ({diffs} diffs)")

    for i, pair in enumerate(data['test']):
        expected = pair.get('output')
        if expected:
            result = solve(pair['input'])
            if result == expected:
                print(f"Test  {i}: PASS")
            else:
                all_pass = False
                diffs = sum(
                    1 for r in range(len(expected))
                    for c in range(len(expected[0]))
                    if result[r][c] != expected[r][c]
                )
                print(f"Test  {i}: FAIL ({diffs} diffs)")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}")
