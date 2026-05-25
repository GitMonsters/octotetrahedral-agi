"""
Puzzle 16b78196: Shape-Through-Notch Interlocking Stacking

Rule: A large block (wall) has notches. Small shapes interlock at notch
locations. For horizontal blocks, shapes are grouped by bounding-box width.
For vertical blocks, plugs uniquely match notch profiles.
"""

import json
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional


def solve(grid: List[List[int]]) -> List[List[int]]:
    H, W = len(grid), len(grid[0])
    flat = [v for row in grid for v in row if v != 0]
    block_color = Counter(flat).most_common(1)[0][0]

    block_cells = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == block_color:
                block_cells.add((r, c))

    block_rows = sorted(set(r for r, c in block_cells))
    block_cols = sorted(set(c for r, c in block_cells))
    is_vertical = (block_rows[-1] - block_rows[0] + 1) > (block_cols[-1] - block_cols[0] + 1)

    shapes = _find_shapes(grid, block_cells, block_color, is_vertical, H, W)
    notches = _find_notches(block_cells, is_vertical, H, W)
    groups = _form_groups_and_place(shapes, notches, is_vertical, block_cells, H, W)

    out = [row[:] for row in grid]
    for s in shapes:
        for r, c in s['cells']:
            out[r][c] = 0
    for g in groups:
        for idx, plc in g['placements']:
            for (dr, dc), v in shapes[idx]['rel_cells'].items():
                r, c = plc[0] + dr, plc[1] + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r][c] = v
    return out


# ─── Shape detection ───────────────────────────────────────────────

def _find_shapes(grid, block_cells, block_color, is_vertical, H, W):
    visited = set()
    shapes = []
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 0 or v == block_color or (r, c) in visited:
                continue
            comp = []
            q = [(r, c)]
            visited.add((r, c))
            while q:
                cr, cc = q.pop(0)
                comp.append((cr, cc, grid[cr][cc]))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                        if grid[nr][nc] != 0 and grid[nr][nc] != block_color:
                            visited.add((nr, nc))
                            q.append((nr, nc))
            cells = [(r2, c2) for r2, c2, _ in comp]
            rmin, rmax = min(r2 for r2, c2 in cells), max(r2 for r2, c2 in cells)
            cmin, cmax = min(c2 for r2, c2 in cells), max(c2 for r2, c2 in cells)
            if is_vertical:
                bc = [c2 for _, c2 in block_cells]
                side = 'left' if (cmin + cmax) / 2 < (min(bc) + max(bc)) / 2 else 'right'
            else:
                br = [r2 for r2, _ in block_cells]
                side = 'above' if (rmin + rmax) / 2 < (min(br) + max(br)) / 2 else 'below'
            rel = {}
            for r2, c2, v2 in comp:
                rel[(r2 - rmin, c2 - cmin)] = v2
            shapes.append({
                'cells': cells, 'rel_cells': rel,
                'rmin': rmin, 'rmax': rmax, 'cmin': cmin, 'cmax': cmax,
                'width': cmax - cmin + 1, 'height': rmax - rmin + 1,
                'side': side, 'n_cells': len(comp),
            })
    return shapes


# ─── Notch detection ───────────────────────────────────────────────

def _group_contiguous(items):
    if not items:
        return []
    runs, cur = [], [items[0]]
    for i in range(1, len(items)):
        if items[i] == cur[-1] + 1:
            cur.append(items[i])
        else:
            runs.append(cur); cur = [items[i]]
    runs.append(cur)
    return runs


def _find_notches(block_cells, is_vertical, H, W):
    if is_vertical:
        return _find_notches_vertical(block_cells, H, W)
    return _find_notches_horizontal(block_cells, H, W)


def _find_notches_vertical(block_cells, H, W):
    row_ranges = {}
    for r, c in block_cells:
        if r not in row_ranges:
            row_ranges[r] = [c, c]
        row_ranges[r][0] = min(row_ranges[r][0], c)
        row_ranges[r][1] = max(row_ranges[r][1], c)
    all_rows = sorted(row_ranges.keys())
    normal_left = Counter(row_ranges[r][0] for r in all_rows).most_common(1)[0][0]
    normal_right = Counter(row_ranges[r][1] for r in all_rows).most_common(1)[0][0]
    min_left = min(row_ranges[r][0] for r in all_rows)
    max_right = max(row_ranges[r][1] for r in all_rows)

    notches = []
    # Left notches: scan from min_left to normal_left (looking for bounded openings)
    for side, normal, extreme, compare in [
        ('left', normal_left, min_left, lambda rr, col: rr[0] > col),
        ('right', normal_right, max_right, lambda rr, col: rr[1] < col)
    ]:
        all_layers = []
        if side == 'left':
            scan_range = range(normal, extreme - 1, -1)
        else:
            scan_range = range(normal, extreme + 2)
        for col_level in scan_range:
            if side == 'left':
                open_rows = sorted(r for r in all_rows if row_ranges[r][0] > col_level)
            else:
                open_rows = sorted(r for r in all_rows if row_ranges[r][1] < col_level)
            if not open_rows:
                continue
            for run in _group_contiguous(open_rows):
                top_r, bot_r = run[0] - 1, run[-1] + 1
                if top_r in row_ranges and bot_r in row_ranges:
                    if side == 'left':
                        if row_ranges[top_r][0] <= col_level and row_ranges[bot_r][0] <= col_level:
                            # Compute actual depths at each row
                            depths = []
                            for r in run:
                                depths.append(row_ranges[r][0] - normal)
                            all_layers.append((tuple(run), tuple(depths)))
                    else:
                        if row_ranges[top_r][1] >= col_level and row_ranges[bot_r][1] >= col_level:
                            depths = []
                            for r in run:
                                depths.append(normal - row_ranges[r][1])
                            all_layers.append((tuple(run), tuple(depths)))

        # Deduplicate and group
        seen = set()
        for rows_t, depths_t in all_layers:
            if rows_t in seen:
                continue
            seen.add(rows_t)
            notches.append({
                'side': side,
                'rows': list(rows_t),
                'profile': list(depths_t),
            })

    return notches


def _find_notches_horizontal(block_cells, H, W):
    col_ranges = {}
    for r, c in block_cells:
        if c not in col_ranges:
            col_ranges[c] = [r, r]
        col_ranges[c][0] = min(col_ranges[c][0], r)
        col_ranges[c][1] = max(col_ranges[c][1], r)
    normal_top = Counter(v[0] for v in col_ranges.values()).most_common(1)[0][0]
    normal_bot = Counter(v[1] for v in col_ranges.values()).most_common(1)[0][0]
    max_bot = max(v[1] for v in col_ranges.values())
    min_top = min(v[0] for v in col_ranges.values())
    min_c = min(col_ranges.keys())
    max_c = max(col_ranges.keys())

    notches = []

    for side, scan_start, scan_end in [
        ('bottom', max_bot, min_top - 1),   # scan entire block depth
        ('top', min_top, max_bot + 1),
    ]:
        layers = []
        if side == 'bottom':
            row_range = range(scan_start, scan_end, -1)
        else:
            row_range = range(scan_start, scan_end)
        for r in row_range:
            block_at_row = set(c for rc, c in block_cells if rc == r)
            open_cols = sorted(c for c in range(min_c, max_c + 1) if c not in block_at_row)
            for run in _group_contiguous(open_cols):
                left, right = run[0] - 1, run[-1] + 1
                if left in block_at_row and right in block_at_row:
                    layers.append((r, tuple(run)))

        # Group connected layers into notches
        if not layers:
            continue
        parent = list(range(len(layers)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b: parent[a] = b

        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                r1, c1 = layers[i]; r2, c2 = layers[j]
                if abs(r1 - r2) <= 1 and set(c1) & set(c2):
                    # Prevent grouping a small hole with a much wider gap
                    if abs(len(c1) - len(c2)) <= 2:
                        union(i, j)

        groups = {}
        for i in range(len(layers)):
            groups.setdefault(find(i), []).append(layers[i])

        for glayers in groups.values():
            rows_in_group = [r for r, _ in glayers]
            # Outermost layer must reach the normal edge for this side
            if side == 'bottom':
                if max(rows_in_group) < normal_bot:
                    continue
                glayers.sort(key=lambda x: x[0])  # innermost (smaller row) first
            else:
                if min(rows_in_group) > normal_top:
                    continue
                glayers.sort(key=lambda x: -x[0])  # innermost (larger row) first
            profile = [len(cols) for _, cols in glayers]
            notches.append({
                'side': side,
                'layers': glayers,
                'profile': profile,
            })

    return notches


# ─── Plug matching ─────────────────────────────────────────────────

def _try_plug_match(si, notch, is_vertical, block_cells):
    if is_vertical:
        return _try_plug_vert(si, notch, block_cells)
    return _try_plug_horiz(si, notch, block_cells)


def _try_plug_vert(si, notch, block_cells):
    """Vertical block: shape's column edge enters notch rows."""
    rows = notch['rows']
    profile = notch['profile']
    n = len(rows)
    sh = si['height']
    rel = si['rel_cells']
    if n > sh:
        return None

    row_ranges = {}
    for r, c in block_cells:
        if r not in row_ranges:
            row_ranges[r] = [c, c]
        row_ranges[r][0] = min(row_ranges[r][0], c)
        row_ranges[r][1] = max(row_ranges[r][1], c)
    normal_left = Counter(row_ranges[r][0] for r in sorted(row_ranges)).most_common(1)[0][0]
    normal_right = Counter(row_ranges[r][1] for r in sorted(row_ranges)).most_common(1)[0][0]

    if notch['side'] == 'left':
        rightmost = {}
        for (dr, dc), v in rel.items():
            if dr not in rightmost or dc > rightmost[dr]:
                rightmost[dr] = dc
        for off in range(sh - n + 1):
            cmins = []; ok = True
            for i in range(n):
                sr = off + i
                if sr not in rightmost:
                    ok = False; break
                free_rightmost = row_ranges[rows[i]][0] - 1
                cmin_candidate = free_rightmost - rightmost[sr]
                cmins.append(cmin_candidate)
            if not ok or not cmins or len(set(cmins)) != 1:
                continue
            cmin = cmins[0]
            rmin = rows[0] - off
            if _no_block_overlap(rel, rmin, cmin, block_cells):
                return (rmin, cmin)

    elif notch['side'] == 'right':
        leftmost = {}
        for (dr, dc), v in rel.items():
            if dr not in leftmost or dc < leftmost[dr]:
                leftmost[dr] = dc
        for off in range(sh - n + 1):
            cmins = []; ok = True
            for i in range(n):
                sr = off + i
                if sr not in leftmost:
                    ok = False; break
                free_leftmost = row_ranges[rows[i]][1] + 1
                cmin_candidate = free_leftmost - leftmost[sr]
                cmins.append(cmin_candidate)
            if not ok or not cmins or len(set(cmins)) != 1:
                continue
            cmin = cmins[0]
            rmin = rows[0] - off
            if _no_block_overlap(rel, rmin, cmin, block_cells):
                return (rmin, cmin)

    return None


def _try_plug_horiz(si, notch, block_cells):
    """Horizontal block: shape's row edge enters notch cols. Must fill ALL layers."""
    layers = notch['layers']  # [(row, cols_tuple), ...] innermost first
    n = len(layers)
    rel = si['rel_cells']
    row_cells = {}
    for (dr, dc), v in rel.items():
        row_cells.setdefault(dr, []).append(dc)
    sh = si['height']
    if n > sh:
        return None

    if notch['side'] == 'bottom':
        # Shape's TOP rows enter notch (row 0 = innermost)
        cmins = []; ok = True
        for i in range(n):
            sr = i
            if sr not in row_cells:
                ok = False; break
            scols = sorted(row_cells[sr])
            ncols = sorted(layers[i][1])
            if len(scols) != len(ncols):
                ok = False; break
            cmins.append(min(ncols) - min(scols))
        if not ok or not cmins or len(set(cmins)) != 1:
            return None
        cmin = cmins[0]
        rmin = layers[0][0]  # shape row 0 at innermost row
        if _no_block_overlap(rel, rmin, cmin, block_cells):
            return (rmin, cmin)

    elif notch['side'] == 'top':
        # Shape's BOTTOM rows enter notch (last row = innermost)
        cmins = []; ok = True
        for i in range(n):
            sr = sh - 1 - i
            if sr not in row_cells:
                ok = False; break
            scols = sorted(row_cells[sr])
            ncols = sorted(layers[i][1])
            if len(scols) != len(ncols):
                ok = False; break
            cmins.append(min(ncols) - min(scols))
        if not ok or not cmins or len(set(cmins)) != 1:
            return None
        cmin = cmins[0]
        rmin = layers[0][0] - (sh - 1)
        if _no_block_overlap(rel, rmin, cmin, block_cells):
            return (rmin, cmin)

    return None


def _no_block_overlap(rel_cells, rmin, cmin, block_cells):
    for (dr, dc) in rel_cells:
        if (rmin + dr, cmin + dc) in block_cells:
            return False
    return True


# ─── Tetris stacking ──────────────────────────────────────────────

def _tetris_stack(si, anchor_dim, notch_side, placed, block_cells, is_vertical, H=30, W=30):
    """Push shape toward block until it touches placed cells or block."""
    rel = si['rel_cells']
    if is_vertical:
        rmin_fixed = anchor_dim
        if notch_side == 'left':
            # Shape on left of block, push rightward toward block
            for cmin in range(-W, W + 1):
                if _no_conflict(rel, rmin_fixed, cmin, placed, block_cells, H, W):
                    if not _no_conflict(rel, rmin_fixed, cmin + 1, placed, block_cells, H, W):
                        return (rmin_fixed, cmin)
        else:
            # Shape on right of block, push leftward toward block
            for cmin in range(W, -W - 1, -1):
                if _no_conflict(rel, rmin_fixed, cmin, placed, block_cells, H, W):
                    if not _no_conflict(rel, rmin_fixed, cmin - 1, placed, block_cells, H, W):
                        return (rmin_fixed, cmin)
    else:
        cmin_fixed = anchor_dim
        if notch_side == 'bottom':
            for rmin in range(H, -H - 1, -1):
                if _no_conflict(rel, rmin, cmin_fixed, placed, block_cells, H, W):
                    if not _no_conflict(rel, rmin - 1, cmin_fixed, placed, block_cells, H, W):
                        return (rmin, cmin_fixed)
        else:
            for rmin in range(-H, H + 1):
                if _no_conflict(rel, rmin, cmin_fixed, placed, block_cells, H, W):
                    if not _no_conflict(rel, rmin + 1, cmin_fixed, placed, block_cells, H, W):
                        return (rmin, cmin_fixed)
    return None


def _no_conflict(rel, rmin, cmin, placed, block, H=30, W=30):
    for (dr, dc) in rel:
        r, c = rmin + dr, cmin + dc
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if (r, c) in placed or (r, c) in block:
            return False
    return True


# ─── Group formation and placement ────────────────────────────────

def _form_groups_and_place(shapes, notches, is_vertical, block_cells, H, W):
    if not is_vertical:
        return _form_groups_horizontal(shapes, notches, block_cells, H, W)
    return _form_groups_vertical(shapes, notches, block_cells, H, W)


def _form_groups_horizontal(shapes, notches, block_cells, H, W):
    """Group shapes by bounding-box width, find notch per group."""
    by_width = {}
    for i, s in enumerate(shapes):
        by_width.setdefault(s['width'], []).append(i)

    sorted_notches = sorted(notches, key=lambda n: sum(n['profile']), reverse=True)
    used_notches = set()
    groups = []

    for w, indices in sorted(by_width.items()):
        plug_idx = None
        plug_plc = None
        matched_notch = None

        for ni, notch in enumerate(sorted_notches):
            if id(notch) in used_notches:
                continue
            opp_side = 'below' if notch['side'] == 'top' else 'above'
            opp_shapes = [i for i in indices if shapes[i]['side'] == opp_side]
            for i in opp_shapes:
                plc = _try_plug_horiz(shapes[i], notch, block_cells)
                if plc is not None:
                    plug_idx = i
                    plug_plc = plc
                    matched_notch = notch
                    break
            if plug_idx is not None:
                break

        if plug_idx is None:
            continue

        used_notches.add(id(matched_notch))

        placements = [(plug_idx, plug_plc)]
        placed = set()
        for (dr, dc) in shapes[plug_idx]['rel_cells']:
            placed.add((plug_plc[0] + dr, plug_plc[1] + dc))

        opp_side = 'below' if matched_notch['side'] == 'top' else 'above'
        remaining = [i for i in indices if i != plug_idx]
        opposite = sorted([i for i in remaining if shapes[i]['side'] == opp_side],
                         key=lambda i: -shapes[i]['n_cells'])
        same = sorted([i for i in remaining if shapes[i]['side'] != opp_side],
                     key=lambda i: -shapes[i]['n_cells'])

        anchor = plug_plc[1]
        for candidates in [opposite, same]:
            for i in candidates:
                plc = _tetris_stack(shapes[i], anchor, matched_notch['side'],
                                   placed, block_cells, False, H, W)
                if plc is not None:
                    placements.append((i, plc))
                    for (dr, dc) in shapes[i]['rel_cells']:
                        placed.add((plc[0] + dr, plc[1] + dc))

        groups.append({'placements': placements})

    return groups


def _form_groups_vertical(shapes, notches, block_cells, H, W):
    """For vertical blocks: find plugs, then assign remaining shapes to closest group."""
    # Step 1: Find plug for each notch
    used_shapes = set()
    notch_groups = []
    sorted_notches = sorted(notches, key=lambda n: sum(n['profile']), reverse=True)

    used_sides = set()
    for notch in sorted_notches:
        if notch['side'] in used_sides:
            continue
        plug_idx = None
        plug_plc = None
        for i, s in enumerate(shapes):
            if i in used_shapes:
                continue
            plc = _try_plug_vert(s, notch, block_cells)
            if plc is not None:
                plug_idx = i
                plug_plc = plc
                break
        if plug_idx is None:
            continue
        used_shapes.add(plug_idx)
        used_sides.add(notch['side'])
        notch_groups.append((notch, plug_idx, plug_plc))

    if not notch_groups:
        return []

    # Compute block edges for distance calculation
    row_ranges = {}
    for r, c in block_cells:
        if r not in row_ranges:
            row_ranges[r] = [c, c]
        row_ranges[r][0] = min(row_ranges[r][0], c)
        row_ranges[r][1] = max(row_ranges[r][1], c)
    normal_left = Counter(row_ranges[r][0] for r in sorted(row_ranges)).most_common(1)[0][0]
    normal_right = Counter(row_ranges[r][1] for r in sorted(row_ranges)).most_common(1)[0][0]

    # Step 2: Build group data
    group_data = []
    for notch, plug_idx, plug_plc in notch_groups:
        placed = set()
        for (dr, dc) in shapes[plug_idx]['rel_cells']:
            placed.add((plug_plc[0] + dr, plug_plc[1] + dc))
        group_data.append({
            'notch': notch,
            'anchor': plug_plc[0],
            'placed': placed,
            'placements': [(plug_idx, plug_plc)],
        })

    # Step 3: Assign remaining shapes one at a time, largest first
    remaining = sorted([i for i in range(len(shapes)) if i not in used_shapes],
                      key=lambda i: -shapes[i]['n_cells'])

    for si in remaining:
        best_group = None
        best_dist = float('inf')
        best_plc = None

        for gi, gd in enumerate(group_data):
            plc = _tetris_stack(shapes[si], gd['anchor'], gd['notch']['side'],
                               gd['placed'], block_cells, True, H, W)
            if plc is None:
                continue
            if gd['notch']['side'] == 'left':
                max_c = max(plc[1] + dc for (dr, dc) in shapes[si]['rel_cells'])
                dist = normal_left - 1 - max_c
            else:
                min_c = min(plc[1] + dc for (dr, dc) in shapes[si]['rel_cells'])
                dist = min_c - normal_right - 1
            if dist < 0:
                continue
            # Prefer closer to block; on tie, prefer matching original side
            if dist < best_dist or (dist == best_dist and best_group is not None and
                    shapes[si]['side'] == gd['notch']['side']):
                best_dist = dist
                best_group = gi
                best_plc = plc

        if best_group is not None:
            gd = group_data[best_group]
            gd['placements'].append((si, best_plc))
            for (dr, dc) in shapes[si]['rel_cells']:
                gd['placed'].add((best_plc[0] + dr, best_plc[1] + dc))
            used_shapes.add(si)

    return [{'placements': gd['placements']} for gd in group_data]


# ─── Validation ────────────────────────────────────────────────────

def validate():
    with open('/Users/evanpieser/arc-puzzle-catalog/dataset/tasks/16b78196.json') as f:
        data = json.load(f)

    all_pass = True
    for idx, pair in enumerate(data['train']):
        result = solve(pair['input'])
        expected = pair['output']
        diffs = []
        for r in range(len(expected)):
            for c in range(len(expected[0])):
                if result[r][c] != expected[r][c]:
                    diffs.append((r, c, result[r][c], expected[r][c]))
        status = "✅ PASS" if not diffs else f"❌ FAIL ({len(diffs)} diffs)"
        print(f"Train {idx}: {status}")
        if diffs:
            all_pass = False
            for r, c, got, exp in diffs[:15]:
                print(f"  ({r},{c}): got {got}, expected {exp}")

    for idx, pair in enumerate(data['test']):
        result = solve(pair['input'])
        if 'output' in pair:
            expected = pair['output']
            diffs = sum(1 for r in range(len(expected)) for c in range(len(expected[0]))
                       if result[r][c] != expected[r][c])
            status = "✅ PASS" if diffs == 0 else f"❌ FAIL ({diffs} diffs)"
        else:
            status = "output generated"
        print(f"Test {idx}: {status}")

    return all_pass


if __name__ == '__main__':
    validate()
