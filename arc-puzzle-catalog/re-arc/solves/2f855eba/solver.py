def transform(input_grid):
    H = len(input_grid)
    W = len(input_grid[0])

    # Find background color (most common)
    color_count = {}
    for r in range(H):
        for c in range(W):
            v = input_grid[r][c]
            color_count[v] = color_count.get(v, 0) + 1
    bg = max(color_count, key=color_count.get)

    # Find connected components of non-bg colors
    visited = set()
    components = []
    for r in range(H):
        for c in range(W):
            if (r, c) in visited or input_grid[r][c] == bg:
                continue
            color = input_grid[r][c]
            comp = set()
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) in visited or cr < 0 or cr >= H or cc < 0 or cc >= W:
                    continue
                if input_grid[cr][cc] != color:
                    continue
                visited.add((cr, cc))
                comp.add((cr, cc))
                stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            components.append({
                'color': color,
                'cells': comp,
                'r1': min(rs), 'c1': min(cs),
                'r2': max(rs), 'c2': max(cs),
                'h': max(rs)-min(rs)+1, 'w': max(cs)-min(cs)+1,
                'count': len(comp),
                'solid': len(comp) == (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
            })

    # Group by color
    by_color = {}
    for comp in components:
        by_color.setdefault(comp['color'], []).append(comp)

    non_bg_colors = [c for c in by_color if c != bg]

    # Find the frame: non-solid component with a rectangular hole
    frame_comp = None
    frame_color = None
    for color in non_bg_colors:
        for comp in by_color[color]:
            if not comp['solid'] and comp['count'] > 10:
                # Check for rectangular hole
                r1, c1, r2, c2 = comp['r1'], comp['c1'], comp['r2'], comp['c2']
                hole_cells = []
                for r in range(r1, r2+1):
                    for c in range(c1, c2+1):
                        if (r, c) not in comp['cells']:
                            hole_cells.append((r, c))
                if hole_cells:
                    hrs = [r for r, c in hole_cells]
                    hcs = [c for r, c in hole_cells]
                    hr1, hc1, hr2, hc2 = min(hrs), min(hcs), max(hrs), max(hcs)
                    hh = hr2 - hr1 + 1
                    hw = hc2 - hc1 + 1
                    if len(hole_cells) == hh * hw:
                        # Valid rectangular hole
                        if frame_comp is None or comp['count'] > frame_comp['count']:
                            frame_comp = comp
                            frame_color = color
                            frame_comp['hole_r1'] = hr1
                            frame_comp['hole_c1'] = hc1
                            frame_comp['hole_r2'] = hr2
                            frame_comp['hole_c2'] = hc2
                            frame_comp['hole_h'] = hh
                            frame_comp['hole_w'] = hw
                            hole_color = input_grid[hr1][hc1]
                            frame_comp['hole_color'] = hole_color

    # Case C: Two components of same non-bg color (cross + detached)
    if frame_comp is None:
        for color in non_bg_colors:
            if len(by_color[color]) == 2:
                return handle_two_components(input_grid, bg, by_color[color], H, W)
        return input_grid

    # Compute borders
    t = frame_comp['hole_r1'] - frame_comp['r1']
    b = frame_comp['r2'] - frame_comp['hole_r2']
    l = frame_comp['hole_c1'] - frame_comp['c1']
    ri = frame_comp['c2'] - frame_comp['hole_c2']
    tb = max(t, b)
    lr = max(l, ri)

    hole_color = frame_comp['hole_color']
    hole_h = frame_comp['hole_h']
    hole_w = frame_comp['hole_w']

    # Find blocks of hole color (not bg)
    target_blocks = []
    if hole_color != bg:
        for comp in by_color.get(hole_color, []):
            if comp['solid']:
                target_blocks.append(comp)

    # Separate into "current hole block" and "other blocks"
    cur_hole = (frame_comp['hole_r1'], frame_comp['hole_c1'],
                frame_comp['hole_r2'], frame_comp['hole_c2'])
    other_blocks = []
    for blk in target_blocks:
        if (blk['r1'], blk['c1'], blk['r2'], blk['c2']) != cur_hole:
            other_blocks.append(blk)

    if other_blocks:
        # Case B: Move frame to wrap another block
        # Choose: minimum total clipping, break ties by distance to TL
        best = None
        best_key = None
        for blk in other_blocks:
            tr1, tc1, tr2, tc2 = blk['r1'], blk['c1'], blk['r2'], blk['c2']
            fr1 = tr1 - tb
            fc1 = tc1 - lr
            fr2 = tr2 + tb
            fc2 = tc2 + lr
            top_clip = max(0, -fr1)
            bot_clip = max(0, fr2 - (H-1))
            left_clip = max(0, -fc1)
            right_clip = max(0, fc2 - (W-1))
            total_clip = top_clip + bot_clip + left_clip + right_clip
            dist_tl = tr1**2 + tc1**2
            key = (total_clip, dist_tl)
            if best_key is None or key < best_key:
                best_key = key
                best = blk
        target = best
        return build_output_with_blocks(input_grid, bg, frame_color, frame_comp,
                                        target, tb, lr, target_blocks, H, W)
    else:
        # Case A: No external blocks (hole color = bg). Use virtual positions.
        return handle_virtual_positions(input_grid, bg, frame_color, frame_comp,
                                        tb, lr, hole_h, hole_w, H, W)


def build_output_with_blocks(grid, bg, frame_color, frame_comp, target, tb, lr,
                              all_blocks, H, W):
    output = [[bg] * W for _ in range(H)]

    # Place all blocks of hole color at original positions
    for blk in all_blocks:
        for r, c in blk['cells']:
            output[r][c] = blk['color']

    # Draw frame around target block
    tr1, tc1, tr2, tc2 = target['r1'], target['c1'], target['r2'], target['c2']
    fr1 = max(0, tr1 - tb)
    fc1 = max(0, tc1 - lr)
    fr2 = min(H-1, tr2 + tb)
    fc2 = min(W-1, tc2 + lr)

    for r in range(fr1, fr2+1):
        for c in range(fc1, fc2+1):
            if not (tr1 <= r <= tr2 and tc1 <= c <= tc2):
                if output[r][c] == bg:
                    output[r][c] = frame_color

    return output


def handle_virtual_positions(grid, bg, frame_color, frame_comp, tb, lr,
                              hole_h, hole_w, H, W):
    cur_r = frame_comp['hole_r1']
    cur_c = frame_comp['hole_c1']
    row_sp = tb + hole_h
    col_sp = lr + hole_w

    # Generate all virtual positions where hole fits in grid
    candidates = []
    for dr in range(-20, 21):
        for dc in range(-20, 21):
            r = cur_r + dr * row_sp
            c = cur_c + dc * col_sp
            if r == cur_r and c == cur_c:
                continue
            # Hole must be fully in grid
            if not (0 <= r and r + hole_h - 1 < H and 0 <= c and c + hole_w - 1 < W):
                continue
            # Compute clipping
            fr1 = r - tb
            fc1 = c - lr
            fr2 = r + hole_h - 1 + tb
            fc2 = c + hole_w - 1 + lr
            top_clip = max(0, -fr1)
            bot_clip = max(0, fr2 - (H-1))
            left_clip = max(0, -fc1)
            right_clip = max(0, fc2 - (W-1))
            total_clip = top_clip + bot_clip + left_clip + right_clip
            n_edges = sum(1 for x in [top_clip, bot_clip, left_clip, right_clip] if x > 0)
            # Frame must clip at exactly one edge
            if n_edges != 1:
                continue
            dist_tl = r**2 + c**2
            manhattan = abs(r - cur_r) + abs(c - cur_c)
            candidates.append((total_clip, manhattan, dist_tl, r, c))

    # Sort: minimum total_clip, then minimum manhattan, then minimum dist_tl
    candidates.sort()

    # Filter: only positions with at least some clipping (edges touching)
    # Actually, pick the absolute minimum total_clip > 0 (or if 0 exists, pick it)
    if not candidates:
        return grid

    target_r, target_c = candidates[0][3], candidates[0][4]

    # Build output
    output = [[bg] * W for _ in range(H)]
    fr1 = max(0, target_r - tb)
    fc1 = max(0, target_c - lr)
    fr2 = min(H-1, target_r + hole_h - 1 + tb)
    fc2 = min(W-1, target_c + hole_w - 1 + lr)

    for r in range(fr1, fr2+1):
        for c in range(fc1, fc2+1):
            if not (target_r <= r <= target_r + hole_h - 1 and
                    target_c <= c <= target_c + hole_w - 1):
                output[r][c] = frame_color

    return output


def handle_two_components(grid, bg, comps, H, W):
    color = comps[0]['color']
    main = max(comps, key=lambda c: c['count'])
    detached = min(comps, key=lambda c: c['count'])

    # Determine relative position of detached to main
    if detached['r2'] < main['r1']:
        # Detached is above main
        gap = main['r1'] - detached['r2'] - 1
        shift = -(detached['h'] + gap)
        axis = 'row'
        direction = 'above'
    elif detached['r1'] > main['r2']:
        # Detached is below main
        gap = detached['r1'] - main['r2'] - 1
        shift = detached['h'] + gap
        axis = 'row'
        direction = 'below'
    elif detached['c2'] < main['c1']:
        # Detached is left of main
        gap = main['c1'] - detached['c2'] - 1
        shift = -(detached['w'] + gap)
        axis = 'col'
        direction = 'left'
    else:
        # Detached is right of main
        gap = detached['c1'] - main['c2'] - 1
        shift = detached['w'] + gap
        axis = 'col'
        direction = 'right'

    output = [[bg] * W for _ in range(H)]

    # Shift main component
    for r, c in main['cells']:
        if axis == 'row':
            nr, nc = r + shift, c
        else:
            nr, nc = r, c + shift
        if 0 <= nr < H and 0 <= nc < W:
            output[nr][nc] = color

    # Place new detached on opposite side
    if axis == 'row':
        if direction == 'above':
            new_main_far = main['r2'] + shift
            new_det_r1 = new_main_far + gap + 1
        else:
            new_main_far = main['r1'] + shift
            new_det_r1 = new_main_far - gap - detached['h']
        for r, c in detached['cells']:
            nr = new_det_r1 + (r - detached['r1'])
            nc = c
            if 0 <= nr < H and 0 <= nc < W:
                output[nr][nc] = color
    else:
        if direction == 'left':
            new_main_far = main['c2'] + shift
            new_det_c1 = new_main_far + gap + 1
        else:
            new_main_far = main['c1'] + shift
            new_det_c1 = new_main_far - gap - detached['w']
        for r, c in detached['cells']:
            nr = r
            nc = new_det_c1 + (c - detached['c1'])
            if 0 <= nr < H and 0 <= nc < W:
                output[nr][nc] = color

    return output
