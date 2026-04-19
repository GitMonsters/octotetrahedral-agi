def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: The input contains two objects:
    1. A rectangular frame made of 2-4 colored line segments (top, bottom, left, right borders)
    2. A separate shape whose bounding box matches the frame's interior dimensions
    
    Output: The frame with the shape placed inside. Each shape cell is colored by the
    nearest border (perpendicular distance). Ties → keep shape color. If the unique
    closest border is missing → cell becomes background.
    """
    from collections import Counter, deque
    from itertools import combinations, product as iterproduct

    rows = len(input_grid)
    cols = len(input_grid[0])

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and not visited[r][c]:
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)

    # Classify each component as a line segment or irregular shape
    line_segments = []  # (comp_index, info_dict)
    other_comps = []    # (comp_index, cells)

    for i, comp in enumerate(components):
        comp_rs = [p[0] for p in comp]
        comp_cs = [p[1] for p in comp]
        min_r, max_r = min(comp_rs), max(comp_rs)
        min_c, max_c = min(comp_cs), max(comp_cs)
        color_set = set(input_grid[r][c] for r, c in comp)
        length = len(comp)

        if len(color_set) != 1 or length < 2:
            other_comps.append((i, comp))
            continue

        color = color_set.pop()
        is_hline = (min_r == max_r and length == max_c - min_c + 1)
        is_vline = (min_c == max_c and length == max_r - min_r + 1)

        if is_hline or is_vline:
            info = {
                'cells': comp, 'color': color,
                'min_r': min_r, 'max_r': max_r,
                'min_c': min_c, 'max_c': max_c,
                'length': length,
                'type': 'h' if is_hline else 'v',
            }
            line_segments.append((i, info))
        else:
            other_comps.append((i, comp))

    # Try to assemble a valid frame from line segments
    best_frame = None

    for n in range(min(4, len(line_segments)), 1, -1):
        if best_frame:
            break
        for seg_pick in combinations(range(len(line_segments)), n):
            if best_frame:
                break
            segs = [line_segments[idx][1] for idx in seg_pick]
            h_segs = [s for s in segs if s['type'] == 'h']
            v_segs = [s for s in segs if s['type'] == 'v']
            if len(h_segs) > 2 or len(v_segs) > 2:
                continue

            for h_roles in iterproduct(['top', 'bottom'], repeat=len(h_segs)):
                if best_frame:
                    break
                if len(set(h_roles)) != len(h_roles):
                    continue
                for v_roles in iterproduct(['left', 'right'], repeat=len(v_segs)):
                    if best_frame:
                        break
                    if len(set(v_roles)) != len(v_roles):
                        continue

                    params = {}
                    valid = True

                    def add_p(key, val):
                        nonlocal valid
                        if key in params:
                            if params[key] != val:
                                valid = False
                        else:
                            params[key] = val

                    for seg, role in zip(h_segs, h_roles):
                        if not valid:
                            break
                        if role == 'top':
                            add_p('top_row', seg['min_r'])
                        else:
                            add_p('bot_row', seg['min_r'])
                        add_p('h_col_start', seg['min_c'])
                        add_p('h_col_end', seg['max_c'])

                    if not valid:
                        continue

                    for seg, role in zip(v_segs, v_roles):
                        if not valid:
                            break
                        if role == 'left':
                            add_p('lft_col', seg['min_c'])
                        else:
                            add_p('rgt_col', seg['min_c'])
                        add_p('v_row_start', seg['min_r'])
                        add_p('v_row_end', seg['max_r'])

                    if not valid:
                        continue

                    # Derive missing params via relationships
                    d = dict(params)
                    for _ in range(10):
                        if 'top_row' in d and 'v_row_start' not in d:
                            d['v_row_start'] = d['top_row'] + 1
                        if 'v_row_start' in d and 'top_row' not in d:
                            d['top_row'] = d['v_row_start'] - 1
                        if 'bot_row' in d and 'v_row_end' not in d:
                            d['v_row_end'] = d['bot_row'] - 1
                        if 'v_row_end' in d and 'bot_row' not in d:
                            d['bot_row'] = d['v_row_end'] + 1
                        if 'lft_col' in d and 'h_col_start' not in d:
                            d['h_col_start'] = d['lft_col'] + 1
                        if 'h_col_start' in d and 'lft_col' not in d:
                            d['lft_col'] = d['h_col_start'] - 1
                        if 'rgt_col' in d and 'h_col_end' not in d:
                            d['h_col_end'] = d['rgt_col'] - 1
                        if 'h_col_end' in d and 'rgt_col' not in d:
                            d['rgt_col'] = d['h_col_end'] + 1

                    needed = ['top_row', 'bot_row', 'lft_col', 'rgt_col']
                    if not all(k in d for k in needed):
                        continue

                    tr, br, lc, rc = d['top_row'], d['bot_row'], d['lft_col'], d['rgt_col']
                    if br <= tr or rc <= lc:
                        continue

                    # Consistency
                    if 'v_row_start' in d and d['v_row_start'] != tr + 1:
                        continue
                    if 'v_row_end' in d and d['v_row_end'] != br - 1:
                        continue
                    if 'h_col_start' in d and d['h_col_start'] != lc + 1:
                        continue
                    if 'h_col_end' in d and d['h_col_end'] != rc - 1:
                        continue

                    H = br - tr - 1
                    W = rc - lc - 1
                    if H <= 0 or W <= 0:
                        continue

                    # Gather non-frame cells as the shape
                    shape_cells = []
                    for ci, comp in other_comps:
                        shape_cells.extend(comp)
                    for idx in range(len(line_segments)):
                        if idx not in seg_pick:
                            shape_cells.extend(line_segments[idx][1]['cells'])

                    if not shape_cells:
                        continue

                    s_min_r = min(r for r, c in shape_cells)
                    s_max_r = max(r for r, c in shape_cells)
                    s_min_c = min(c for r, c in shape_cells)
                    s_max_c = max(c for r, c in shape_cells)

                    if s_max_r - s_min_r + 1 == H and s_max_c - s_min_c + 1 == W:
                        top_seg = bot_seg = lft_seg = rgt_seg = None
                        for seg, role in zip(h_segs, h_roles):
                            if role == 'top':
                                top_seg = seg
                            else:
                                bot_seg = seg
                        for seg, role in zip(v_segs, v_roles):
                            if role == 'left':
                                lft_seg = seg
                            else:
                                rgt_seg = seg

                        best_frame = {
                            'top': top_seg, 'bottom': bot_seg,
                            'left': lft_seg, 'right': rgt_seg,
                            'top_row': tr, 'bot_row': br,
                            'lft_col': lc, 'rgt_col': rc,
                            'H': H, 'W': W,
                            'seg_pick': seg_pick,
                            'shape_cells': shape_cells,
                            's_min_r': s_min_r, 's_min_c': s_min_c,
                        }

    frame = best_frame
    H, W = frame['H'], frame['W']

    # Build shape grid
    shape_grid = [[bg] * W for _ in range(H)]
    for r, c in frame['shape_cells']:
        shape_grid[r - frame['s_min_r']][c - frame['s_min_c']] = input_grid[r][c]

    # Build output
    out_H = H + 2
    out_W = W + 2
    output = [[bg] * out_W for _ in range(out_H)]

    top_color = frame['top']['color'] if frame['top'] else None
    bot_color = frame['bottom']['color'] if frame['bottom'] else None
    lft_color = frame['left']['color'] if frame['left'] else None
    rgt_color = frame['right']['color'] if frame['right'] else None

    # Place border colors
    if top_color is not None:
        for c in range(1, out_W - 1):
            output[0][c] = top_color
    if bot_color is not None:
        for c in range(1, out_W - 1):
            output[out_H - 1][c] = bot_color
    if lft_color is not None:
        for r in range(1, out_H - 1):
            output[r][0] = lft_color
    if rgt_color is not None:
        for r in range(1, out_H - 1):
            output[r][out_W - 1] = rgt_color

    # Fill interior using distance rule
    for ir in range(H):
        for ic in range(W):
            sv = shape_grid[ir][ic]
            if sv == bg:
                continue

            out_r = ir + 1
            out_c = ic + 1

            d_top = out_r
            d_bot = out_H - 1 - out_r
            d_lft = out_c
            d_rgt = out_W - 1 - out_c
            min_d = min(d_top, d_bot, d_lft, d_rgt)

            closest = []
            if d_top == min_d:
                closest.append(top_color)
            if d_bot == min_d:
                closest.append(bot_color)
            if d_lft == min_d:
                closest.append(lft_color)
            if d_rgt == min_d:
                closest.append(rgt_color)

            if len(closest) == 1:
                if closest[0] is not None:
                    output[out_r][out_c] = closest[0]
                # else: unique closest is missing border → stays bg
            else:
                # Tie → keep shape color
                output[out_r][out_c] = sv

    return output
