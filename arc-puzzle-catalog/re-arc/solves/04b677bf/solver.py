def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    H = len(input_grid)
    W = len(input_grid[0])

    # Find background color
    flat = [input_grid[r][c] for r in range(H) for c in range(W)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-bg cells
    non_bg = {}
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != bg:
                non_bg[(r, c)] = input_grid[r][c]

    # 8-connected components
    visited = set()
    components = []
    for (r, c) in non_bg:
        if (r, c) in visited:
            continue
        stack = [(r, c)]
        comp = []
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp.append((cr, cc))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in non_bg and (nr, nc) not in visited:
                        stack.append((nr, nc))
        components.append(sorted(comp))

    # Classify components
    blocks = []
    stamp_comp = None
    isolated = []

    for comp in components:
        if len(comp) == 1:
            isolated.append(comp[0])
            continue
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        rmin, rmax = min(rows), max(rows)
        cmin, cmax = min(cols), max(cols)
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if len(comp) == h * w:
            blocks.append((rmin, cmin, h, w))
        else:
            stamp_comp = comp

    if stamp_comp is None:
        return [row[:] for row in input_grid]

    # Get stamp relative positions and colors
    sr_min = min(r for r, c in stamp_comp)
    sc_min = min(c for r, c in stamp_comp)
    sr_max = max(r for r, c in stamp_comp)
    sc_max = max(c for r, c in stamp_comp)
    sh = sr_max - sr_min + 1
    sw = sc_max - sc_min + 1

    stamp_rel = {}
    for r, c in stamp_comp:
        stamp_rel[(r - sr_min, c - sc_min)] = non_bg[(r, c)]

    # Find foreground color (from blocks or isolated pixels)
    fg_color = None
    for rmin, cmin, bh, bw in blocks:
        fg_color = non_bg[(rmin, cmin)]
        break
    if fg_color is None and isolated:
        fg_color = non_bg[isolated[0]]

    # Determine reference point
    colors_in_stamp = set(stamp_rel.values())

    if len(colors_in_stamp) > 1:
        # Mixed colors: reference is where fg_color is
        ref = None
        for pos, col in stamp_rel.items():
            if col == fg_color:
                ref = pos
                break
    else:
        # Single color: reference is corner opposite the missing corner
        corners = [(0, 0), (0, sw - 1), (sh - 1, 0), (sh - 1, sw - 1)]
        ref = None
        for corner in corners:
            if corner not in stamp_rel:
                ref = (sh - 1 - corner[0], sw - 1 - corner[1])
                break

    if ref is None:
        return [row[:] for row in input_grid]

    # Build output
    output = [row[:] for row in input_grid]

    # Apply stamp to isolated pixels (scale=1)
    for pr, pc in isolated:
        for (sr, sc), col in stamp_rel.items():
            nr = pr + sr - ref[0]
            nc = pc + sc - ref[1]
            if 0 <= nr < H and 0 <= nc < W:
                output[nr][nc] = col

    # Apply stamp to rectangular blocks (scaled)
    for rmin, cmin, bh, bw in blocks:
        origin_r = rmin - ref[0] * bh
        origin_c = cmin - ref[1] * bw
        for (sr, sc), col in stamp_rel.items():
            for r in range(origin_r + sr * bh, origin_r + (sr + 1) * bh):
                for c in range(origin_c + sc * bw, origin_c + (sc + 1) * bw):
                    if 0 <= r < H and 0 <= c < W:
                        output[r][c] = col

    return output
