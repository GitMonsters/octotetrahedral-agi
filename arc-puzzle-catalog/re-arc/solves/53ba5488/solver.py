from collections import Counter


def transform(grid):
    H, W = len(grid), len(grid[0])
    bg = Counter(c for row in grid for c in row).most_common(1)[0][0]
    non_bg = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not non_bg:
        return [row[:] for row in grid]
    fill = Counter(v for _, _, v in non_bg).most_common(1)[0][0]
    fill_cells = set((r, c) for r, c, v in non_bg if v == fill)

    # Find largest connected component of fill cells (the block)
    visited, best_comp = set(), []
    for sr, sc in fill_cells:
        if (sr, sc) in visited:
            continue
        comp, stack = [], [(sr, sc)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited or (cr, cc) not in fill_cells:
                continue
            visited.add((cr, cc))
            comp.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (cr + dr, cc + dc) in fill_cells and (cr + dr, cc + dc) not in visited:
                    stack.append((cr + dr, cc + dc))
        if len(comp) > len(best_comp):
            best_comp = comp

    block = set(best_comp)
    r1 = min(r for r, c in block)
    r2 = max(r for r, c in block)
    c1 = min(c for r, c in block)
    c2 = max(c for r, c in block)

    block_cells = set((r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1))
    markers = [(r, c, v) for r, c, v in non_bg if (r, c) not in block_cells]

    out = [row[:] for row in grid]

    # Project each marker onto the nearest block edge
    proj = {}
    for r, c, v in markers:
        if r1 <= r <= r2:
            if c < c1:
                proj[(r, c1)] = v
            elif c > c2:
                proj[(r, c2)] = v
        elif c1 <= c <= c2:
            if r < r1:
                proj[(r1, c)] = v
            elif r > r2:
                proj[(r2, c)] = v

    for (r, c), v in proj.items():
        out[r][c] = v

    # Secondary rule: when there are no top-side markers, apply border bg pattern.
    # Top markers = markers above the block projecting onto the top row.
    top_markers = [(r, c, v) for r, c, v in markers if r < r1 and c1 <= c <= c2]
    if top_markers:
        return out

    # Build clockwise border with edge labels: T=top, R=right, B=bottom, L=left
    border = []
    for c in range(c1, c2 + 1):
        border.append((r1, c, 'T'))
    for r in range(r1 + 1, r2 + 1):
        border.append((r, c2, 'R'))
    for c in range(c2 - 1, c1 - 1, -1):
        border.append((r2, c, 'B'))
    for r in range(r2 - 1, r1, -1):
        border.append((r, c1, 'L'))
    n = len(border)

    # "Real" border markers: cells whose projected value differs from fill
    real_marker_set = {pos for pos, v in proj.items() if v != fill}

    # Apply border rule to each non-marker fill cell
    for i, (r, c, edge) in enumerate(border):
        if (r, c) in real_marker_set or out[r][c] != fill:
            continue
        # Distance to previous clockwise real marker
        dp = next(
            d for d in range(1, n + 1)
            if (border[(i - d) % n][0], border[(i - d) % n][1]) in real_marker_set
        )
        if edge == 'B':
            # Bottom row always stays fill
            pass
        elif edge == 'T':
            # Top row: alternating — even distance from prev marker → bg
            if dp % 2 == 0:
                out[r][c] = bg
        else:
            # Left/right cols: first cell after marker = fill, rest = bg
            if dp > 1:
                out[r][c] = bg

    return out
