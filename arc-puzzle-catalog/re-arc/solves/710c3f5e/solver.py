"""Solver for ARC puzzle 710c3f5e.

Rule: Input splits into canvas+dict halves. Dict has rooms (connected components
of non-bg cells). Rooms matched by marker shape to canvas clusters get stamped.
All-fill rooms match visible clusters of their fill color.
Invisible-marker rooms in the first column band of the dict grid get stamped
using a symmetry rule.
"""
from collections import Counter, deque


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    rows = len(input_grid)
    cols = len(input_grid[0])

    def get_bg(grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def find_rooms(grid, bg):
        R, C = len(grid), len(grid[0])
        vis = set()
        rooms = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] != bg and (r, c) not in vis:
                    comp = []
                    q = deque([(r, c)])
                    vis.add((r, c))
                    while q:
                        cr, cc = q.popleft()
                        comp.append((cr, cc, grid[cr][cc]))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in vis and grid[nr][nc] != bg:
                                vis.add((nr, nc))
                                q.append((nr, nc))
                    rooms.append(comp)
        return rooms

    def stamp_room(output, cells, r0, c0, sr, sc, OR, OC):
        for r, c, v in cells:
            nr = r - r0 + sr
            nc = c - c0 + sc
            if 0 <= nr < OR and 0 <= nc < OC:
                output[nr][nc] = v

    # Determine split
    for split_type in ['vertical', 'horizontal']:
        if split_type == 'vertical' and cols % 2 != 0:
            continue
        if split_type == 'horizontal' and rows % 2 != 0:
            continue

        if split_type == 'vertical':
            half = cols // 2
            ha = [row[:half] for row in input_grid]
            hb = [row[half:] for row in input_grid]
        else:
            half = rows // 2
            ha = input_grid[:half]
            hb = input_grid[half:]

        bga, bgb = get_bg(ha), get_bg(hb)
        if bga == bgb:
            continue
        na = sum(1 for row in ha for c in row if c != bga)
        nb = sum(1 for row in hb for c in row if c != bgb)
        canvas, dct = (ha, hb) if na <= nb else (hb, ha)
        cbg = get_bg(canvas)
        dbg = get_bg(dct)
        OR, OC = len(canvas), len(canvas[0])

        rooms = find_rooms(dct, dbg)
        infos = []
        for room in rooms:
            fill = Counter(v for r, c, v in room).most_common(1)[0][0]
            markers = {}
            for r, c, v in room:
                if v != fill:
                    markers.setdefault(v, []).append((r, c))
            rs = [r for r, c, v in room]
            cs = [c for r, c, v in room]
            infos.append({
                'cells': room, 'fill': fill, 'markers': markers,
                'r0': min(rs), 'c0': min(cs),
                'h': max(rs) - min(rs) + 1, 'w': max(cs) - min(cs) + 1,
            })

        # Canvas non-bg by color
        cbc = {}
        for r in range(OR):
            for c in range(OC):
                v = canvas[r][c]
                if v != cbg:
                    cbc.setdefault(v, []).append((r, c))

        output = [row[:] for row in canvas]
        stamped = set()

        # === PHASE 1: Match rooms by marker shape ===
        for color in sorted(cbc.keys()):
            remaining = set(cbc[color])
            cands = [ri for ri, info in enumerate(infos)
                     if ri not in stamped and color in info['markers']]
            for ri in cands:
                if ri in stamped or not remaining:
                    break
                mpos = infos[ri]['markers'][color]
                tried = set()
                found = False
                for cr, cc in sorted(remaining):
                    for mr, mc in mpos:
                        off = (cr - mr, cc - mc)
                        if off in tried:
                            continue
                        tried.add(off)
                        shifted = {(mr2 + off[0], mc2 + off[1]) for mr2, mc2 in mpos}
                        if shifted.issubset(remaining) and all(
                            0 <= r < OR and 0 <= c < OC and canvas[r][c] == color
                            for r, c in shifted
                        ):
                            sr = infos[ri]['r0'] + off[0]
                            sc = infos[ri]['c0'] + off[1]
                            h, w = infos[ri]['h'], infos[ri]['w']
                            if 0 <= sr and sr + h <= OR and 0 <= sc and sc + w <= OC:
                                stamp_room(output, infos[ri]['cells'], infos[ri]['r0'],
                                           infos[ri]['c0'], sr, sc, OR, OC)
                                remaining -= shifted
                                stamped.add(ri)
                                found = True
                                break
                    if found:
                        break

        # === PHASE 2: All-fill rooms (single visible color) ===
        fill_rooms = [(ri, infos[ri]) for ri in range(len(infos))
                      if ri not in stamped and not infos[ri]['markers']
                      and infos[ri]['fill'] != cbg and infos[ri]['fill'] in cbc]
        fill_rooms.sort(key=lambda x: x[1]['h'] * x[1]['w'], reverse=True)

        for color in set(info['fill'] for _, info in fill_rooms):
            unclaimed = set(cbc.get(color, []))
            for ri, info in fill_rooms:
                if ri in stamped or info['fill'] != color:
                    continue
                h, w = info['h'], info['w']
                best_pos = None
                best_count = 0
                for sr in range(OR - h + 1):
                    for sc in range(OC - w + 1):
                        conflict = False
                        count = 0
                        for r in range(sr, sr + h):
                            for c in range(sc, sc + w):
                                if output[r][c] != cbg and output[r][c] != info['fill']:
                                    conflict = True
                                    break
                                if (r, c) in unclaimed:
                                    count += 1
                            if conflict:
                                break
                        if not conflict and count > best_count:
                            best_count = count
                            best_pos = (sr, sc)
                if best_pos:
                    stamp_room(output, info['cells'], info['r0'], info['c0'],
                               best_pos[0], best_pos[1], OR, OC)
                    stamped.add(ri)
                    for r in range(best_pos[0], best_pos[0] + h):
                        for c in range(best_pos[1], best_pos[1] + w):
                            unclaimed.discard((r, c))

        # === PHASE 3: Invisible-marker rooms ===
        # These have fill != cbg and markers all = cbg (or no markers with non-fill)
        # Only rooms in the FIRST column band of the dict grid get stamped.
        
        # Find dict grid column bands (separated by full-bg columns)
        DR, DC = len(dct), len(dct[0])
        bg_cols = set()
        for c in range(DC):
            if all(dct[r][c] == dbg for r in range(DR)):
                bg_cols.add(c)
        
        col_bands = []
        in_band = False
        bs = 0
        for c in range(DC):
            if c in bg_cols:
                if in_band:
                    col_bands.append((bs, c - 1))
                    in_band = False
            else:
                if not in_band:
                    bs = c
                    in_band = True
        if in_band:
            col_bands.append((bs, DC - 1))
        
        first_band = col_bands[0] if col_bands else None
        
        inv_rooms = []
        for ri, info in enumerate(infos):
            if ri in stamped:
                continue
            if info['fill'] == cbg:
                continue  # Invisible stamps
            has_only_cbg_markers = (
                len(info['markers']) == 0 or
                all(mc == cbg for mc in info['markers'])
            )
            if not has_only_cbg_markers:
                continue
            # Check if in first column band
            if first_band and info['c0'] >= first_band[0] and (info['c0'] + info['w'] - 1) <= first_band[1]:
                inv_rooms.append(ri)

        if len(inv_rooms) == 2:
            # Use symmetry rule: positions sum to (OR+1, OC+1)
            # Plus: first room goes at (max_visible_end_row - 1, OC - room_width)
            
            # Sort by dict row
            inv_rooms.sort(key=lambda ri: infos[ri]['r0'])
            ri0, ri1 = inv_rooms
            
            # Find max visible stamp end row
            max_end_r = -1
            for sri, info in enumerate(infos):
                if sri in stamped:
                    h = info['h']
                    # Need to know the stamp position
                    # Recompute from output: find where the room was stamped
                    for sr_try in range(OR):
                        for sc_try in range(OC):
                            match = True
                            cnt = 0
                            for r, c, v in info['cells']:
                                nr = r - info['r0'] + sr_try
                                nc = c - info['c0'] + sc_try
                                if 0 <= nr < OR and 0 <= nc < OC:
                                    if output[nr][nc] == v:
                                        cnt += 1
                                    else:
                                        match = False
                                        break
                                else:
                                    match = False
                                    break
                            if match and cnt == len(info['cells']):
                                end_r = sr_try + info['h'] - 1
                                if end_r > max_end_r:
                                    max_end_r = end_r
                                break
                        else:
                            continue
                        break
            
            info0 = infos[ri0]
            info1 = infos[ri1]
            
            sr0 = max_end_r - 1 if max_end_r >= 0 else OR // 2
            sc0 = OC - info0['w']
            sr1 = OR + 1 - sr0
            sc1 = OC + 1 - sc0
            
            # Verify positions are valid
            if (0 <= sr0 and sr0 + info0['h'] <= OR and 0 <= sc0 and sc0 + info0['w'] <= OC
                and 0 <= sr1 and sr1 + info1['h'] <= OR and 0 <= sc1 and sc1 + info1['w'] <= OC):
                stamp_room(output, info0['cells'], info0['r0'], info0['c0'],
                           sr0, sc0, OR, OC)
                stamp_room(output, info1['cells'], info1['r0'], info1['c0'],
                           sr1, sc1, OR, OC)
                stamped.add(ri0)
                stamped.add(ri1)
        elif len(inv_rooms) == 1:
            # Single invisible room: try maximize min distance
            ri = inv_rooms[0]
            info = infos[ri]
            h, w = info['h'], info['w']
            nonbg = {(r, c) for r in range(OR) for c in range(OC) if output[r][c] != cbg}
            best_pos = None
            best_score = -1
            for sr in range(OR - h + 1):
                for sc in range(OC - w + 1):
                    valid = True
                    for r, c, v in info['cells']:
                        nr = r - info['r0'] + sr
                        nc = c - info['c0'] + sc
                        if (nr, nc) in nonbg:
                            valid = False
                            break
                    if not valid:
                        continue
                    min_d = min((abs(r - info['r0'] + sr - pr) + abs(c - info['c0'] + sc - pc)
                                 for r, c, v in info['cells'] if v != cbg
                                 for pr, pc in nonbg), default=OR + OC)
                    if min_d > best_score:
                        best_score = min_d
                        best_pos = (sr, sc)
            if best_pos:
                stamp_room(output, info['cells'], info['r0'], info['c0'],
                           best_pos[0], best_pos[1], OR, OC)

        return output

    return input_grid
