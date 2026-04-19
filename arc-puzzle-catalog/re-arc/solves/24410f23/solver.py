import copy
from collections import Counter, deque


def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    cnt = Counter()
    for r in input_grid:
        for v in r:
            cnt[v] += 1
    colors = sorted(cnt.keys())

    # Find solid rectangle (3-color case)
    rect_color = None
    rect_bounds = None
    for v in colors:
        positions = []
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == v:
                    positions.append((r, c))
        if not positions:
            continue
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        expected = (max_r - min_r + 1) * (max_c - min_c + 1)
        if expected == len(positions) and min(max_r - min_r + 1, max_c - min_c + 1) >= 2:
            rect_color = v
            rect_bounds = (min_r, max_r, min_c, max_c)
            break

    if rect_color is not None:
        # 3-color case
        remaining = [v for v in colors if v != rect_color]
        mk_color = min(remaining, key=lambda v: cnt[v])
        rt, rb, rl, rr = rect_bounds
    else:
        # 2-color case
        bg_color = cnt.most_common(1)[0][0]
        mk_color = [v for v in colors if v != bg_color][0]
        rt, rb, rl, rr = _find_invisible_rect(input_grid, rows, cols, mk_color)

    return _apply_fill(input_grid, rt, rb, rl, rr, mk_color)


def _find_invisible_rect(grid, rows, cols, mk):
    """Find invisible rectangle for 2-color grids using block analysis."""
    markers = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == mk:
                markers.add((r, c))

    # Find connected components
    visited = set()
    components = []
    for start in markers:
        if start in visited:
            continue
        comp = set()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            comp.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in markers and (nr, nc) not in visited:
                    queue.append((nr, nc))
        components.append(comp)

    components.sort(key=len, reverse=True)
    block = components[0]

    block_rows = [r for r, c in block]
    block_cols = [c for r, c in block]
    bmin_r, bmax_r = min(block_rows), max(block_rows)
    bmin_c, bmax_c = min(block_cols), max(block_cols)
    bh = bmax_r - bmin_r + 1
    bw = bmax_c - bmin_c + 1

    # Determine which corner the block is closest to
    center_r = (bmin_r + bmax_r) / 2
    center_c = (bmin_c + bmax_c) / 2
    top = center_r < rows / 2
    left = center_c < cols / 2

    rect_width = bw + bh
    rect_height = bw + 2 * bh

    if top and left:
        rr = cols - bw
        rl = rr - rect_width + 1
        rb = rows - 3 * bh
        rt = rb - rect_height + 1
    elif top and not left:
        rl = bw - 1
        rr = rl + rect_width - 1
        rb = rows - 3 * bh
        rt = rb - rect_height + 1
    elif not top and left:
        rr = cols - bw
        rl = rr - rect_width + 1
        rt = 3 * bh - 1
        rb = rt + rect_height - 1
    else:
        rl = bw - 1
        rr = rl + rect_width - 1
        rt = 3 * bh - 1
        rb = rt + rect_height - 1

    # Verify marker-free
    free = all(
        (r, c) not in markers
        for r in range(max(0, rt), min(rows, rb + 1))
        for c in range(max(0, rl), min(cols, rr + 1))
    )

    if free and 0 <= rt and rb < rows and 0 <= rl and rr < cols:
        return (rt, rb, rl, rr)

    # Fallback: brute-force search for any working rect
    return _brute_force_rect(grid, rows, cols, mk, markers)


def _brute_force_rect(grid, rows, cols, mk, markers):
    """Brute-force search for the invisible rect."""
    best = None
    best_score = -1

    for rt in range(rows - 3):
        for rb in range(rt + 3, min(rt + rows // 2 + 1, rows)):
            for rl in range(cols - 3):
                for rr in range(rl + 3, min(rl + cols // 2 + 1, cols)):
                    free = all(
                        (r, c) not in markers
                        for r in range(rt, rb + 1)
                        for c in range(rl, rr + 1)
                    )
                    if not free:
                        continue

                    h, w = rb - rt + 1, rr - rl + 1
                    # Prefer rects with markers on multiple sides
                    sides = 0
                    if any(any((r, c) in markers for c in range(rl)) for r in range(rt, rb + 1)):
                        sides += 1
                    if any(any((r, c) in markers for c in range(rr + 1, cols)) for r in range(rt, rb + 1)):
                        sides += 1
                    if any(any((r, c) in markers for r in range(rt)) for c in range(rl, rr + 1)):
                        sides += 1
                    if any(any((r, c) in markers for r in range(rb + 1, rows)) for c in range(rl, rr + 1)):
                        sides += 1

                    ratio = max(h, w) / min(h, w)
                    score = sides * 1000 - ratio * 100 + h * w

                    if score > best_score:
                        best_score = score
                        best = (rt, rb, rl, rr)

    return best if best else (0, rows - 1, 0, cols - 1)


def _apply_fill(grid, rt, rb, rl, rr, mk):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    for r in range(rt, rb + 1):
        left_markers = [c for c in range(rl) if grid[r][c] == mk]
        if left_markers:
            farthest = min(left_markers)
            for c in range(farthest, rl):
                out[r][c] = mk
        right_markers = [c for c in range(rr + 1, cols) if grid[r][c] == mk]
        if right_markers:
            farthest = max(right_markers)
            for c in range(rr + 1, farthest + 1):
                out[r][c] = mk

    for c in range(rl, rr + 1):
        top_markers = [r for r in range(rt) if grid[r][c] == mk]
        if top_markers:
            farthest = min(top_markers)
            for r in range(farthest, rt):
                out[r][c] = mk
        bot_markers = [r for r in range(rb + 1, rows) if grid[r][c] == mk]
        if bot_markers:
            farthest = max(bot_markers)
            for r in range(rb + 1, farthest + 1):
                out[r][c] = mk

    return out
