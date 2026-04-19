def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])

    # Find background color (most common)
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Collect edge markers (non-bg pixels on grid boundaries)
    top_markers = {c: grid[0][c] for c in range(W) if grid[0][c] != bg}
    bot_markers = {c: grid[H-1][c] for c in range(W) if grid[H-1][c] != bg}
    left_markers = {r: grid[r][0] for r in range(H) if grid[r][0] != bg}
    right_markers = {r: grid[r][W-1] for r in range(H) if grid[r][W-1] != bg}

    # Find rectangle in the interior
    interior_counts = Counter()
    for r in range(1, H-1):
        for c in range(1, W-1):
            if grid[r][c] != bg:
                interior_counts[grid[r][c]] += 1

    rect_color = None
    r1 = r2 = c1 = c2 = None
    for color, cnt in interior_counts.most_common():
        positions = [(r, c) for r in range(1, H-1) for c in range(1, W-1)
                     if grid[r][c] == color]
        if not positions:
            continue
        rmin = min(r for r, c in positions)
        rmax = max(r for r, c in positions)
        cmin = min(c for r, c in positions)
        cmax = max(c for r, c in positions)
        expected = (rmax - rmin + 1) * (cmax - cmin + 1)
        if expected >= cnt and cnt >= 6:
            rect_color = color
            r1, r2, c1, c2 = rmin, rmax, cmin, cmax
            break

    if rect_color is not None:
        return _transform_with_rect(grid, H, W, bg, rect_color,
                                     r1, r2, c1, c2,
                                     top_markers, bot_markers,
                                     left_markers, right_markers)
    else:
        return _transform_no_rect(grid, H, W, bg,
                                   top_markers, bot_markers,
                                   left_markers, right_markers)


def _transform_with_rect(grid, H, W, bg, rc,
                          r1, r2, c1, c2,
                          top_m, bot_m, left_m, right_m):
    # Step 1: Project non-bg markers onto nearest rect edge
    for c, v in top_m.items():
        if c1 <= c <= c2:
            grid[r1][c] = v
    for c, v in bot_m.items():
        if c1 <= c <= c2:
            grid[r2][c] = v
    for r, v in left_m.items():
        if r1 <= r <= r2:
            grid[r][c1] = v
    for r, v in right_m.items():
        if r1 <= r <= r2:
            grid[r][c2] = v

    # Step 2: Check for complete crossings
    # "Real" markers = non-bg AND non-rect-colored
    def real(markers):
        return {k: v for k, v in markers.items() if v != rc}

    real_top = real(top_m)
    real_bot = real(bot_m)
    real_left = real(left_m)
    real_right = real(right_m)

    # Crossing cols: both top and bottom have real markers at same col within rect
    cross_cols = set()
    for c in real_top:
        if c in real_bot and c1 <= c <= c2:
            cross_cols.add(c)

    # Crossing rows: both left and right have real markers at same row within rect
    cross_rows = set()
    for r in real_left:
        if r in real_right and r1 <= r <= r2:
            cross_rows.add(r)

    has_crossing = len(cross_cols) > 0 and len(cross_rows) > 0

    if not has_crossing:
        return grid

    # Compute spans of real markers on each edge (relative to rect)
    def span(markers, lo, hi):
        positions = [k for k in markers if lo <= k <= hi]
        if not positions:
            return None
        return (min(positions), max(positions))

    top_span = span(real_top, c1, c2)
    bot_span = span(real_bot, c1, c2)
    left_span = span(real_left, r1, r2)
    right_span = span(real_right, r1, r2)

    # Step 3: Exit rule - each real marker's line exits on opposite rect edge
    # Exit = bg if: no opposite marker AND outside opposite span
    for c in real_top:
        if c1 <= c <= c2:
            # Top marker exits at bottom edge
            if c not in bot_m:  # no override
                if bot_span and (c < bot_span[0] or c > bot_span[1]):
                    grid[r2][c] = bg
                elif bot_span is None:
                    grid[r2][c] = bg

    for c in real_bot:
        if c1 <= c <= c2:
            if c not in top_m:
                if top_span and (c < top_span[0] or c > top_span[1]):
                    grid[r1][c] = bg
                elif top_span is None:
                    grid[r1][c] = bg

    for r in real_left:
        if r1 <= r <= r2:
            if r not in right_m:
                if right_span and (r < right_span[0] or r > right_span[1]):
                    grid[r][c2] = bg
                elif right_span is None:
                    grid[r][c2] = bg

    for r in real_right:
        if r1 <= r <= r2:
            if r not in left_m:
                if left_span and (r < left_span[0] or r > left_span[1]):
                    grid[r][c1] = bg
                elif left_span is None:
                    grid[r][c1] = bg

    # Step 4: Crossing shadows
    for cc in cross_cols:
        for cr in cross_rows:
            # Top entry at (r1, cc) -> shadow 1 right
            sc = cc + 1
            if sc <= c2 and grid[r1][sc] == rc:
                if top_span and (sc < top_span[0] or sc > top_span[1]):
                    grid[r1][sc] = bg

            # Bottom entry at (r2, cc) -> shadow 1 left
            sc = cc - 1
            if sc >= c1 and grid[r2][sc] == rc:
                if bot_span and (sc < bot_span[0] or sc > bot_span[1]):
                    grid[r2][sc] = bg

            # Left entry at (cr, c1) -> shadow 1 down
            sr = cr + 1
            if sr <= r2 and grid[sr][c1] == rc:
                if left_span and (sr < left_span[0] or sr > left_span[1]):
                    grid[sr][c1] = bg

            # Right entry at (cr, c2) -> shadow 1 up
            sr = cr - 1
            if sr >= r1 and grid[sr][c2] == rc:
                if right_span and (sr < right_span[0] or sr > right_span[1]):
                    grid[sr][c2] = bg

    return grid


def _transform_no_rect(grid, H, W, bg, top_m, bot_m, left_m, right_m):
    # Virtual frame: determine bounds from marker patterns
    # Column range from top/bottom patterns
    all_marked_cols = set(top_m.keys()) | set(bot_m.keys())
    if not all_marked_cols:
        return grid
    col_min = min(all_marked_cols)
    col_max = max(all_marked_cols)

    # Row range from side markers
    all_side_rows = set(left_m.keys()) | set(right_m.keys())
    if not all_side_rows:
        return grid

    side_min = min(all_side_rows)
    side_max = max(all_side_rows)

    # Virtual rect: height = width = col_max - col_min + 1
    width = col_max - col_min + 1
    # Center on side markers, with bias
    n_right = len(right_m)
    n_left = len(left_m)
    vr_top = side_min - n_right
    vr_bot = side_max + n_left

    # Fallback: if computed height doesn't match width, use width-based
    computed_height = vr_bot - vr_top + 1
    if computed_height != width:
        center = (side_min + side_max) / 2.0
        vr_top = int(center - width / 2.0 + 0.5)
        vr_bot = vr_top + width - 1

    # Place top pattern at vr_top row
    for c, v in top_m.items():
        if 0 <= vr_top < H:
            grid[vr_top][c] = v

    # Place bottom pattern at vr_bot row
    for c, v in bot_m.items():
        if 0 <= vr_bot < H:
            grid[vr_bot][c] = v

    # Side markers project to nearest virtual rect edge
    for r, v in left_m.items():
        if vr_top <= r <= vr_bot:
            grid[r][col_min] = v
    for r, v in right_m.items():
        if vr_top <= r <= vr_bot:
            grid[r][col_max] = v

    return grid
