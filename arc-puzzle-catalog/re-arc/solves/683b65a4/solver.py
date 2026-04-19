from collections import Counter


def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    bg = Counter(grid[r][c] for r in range(H) for c in range(W)).most_common(1)[0][0]

    rect = _find_solid_rect(grid, H, W, bg)
    if rect is None:
        rect = _find_rect_from_pattern(grid, H, W, bg)
    if rect is None:
        return out

    rmin, rmax, cmin, cmax = rect

    # Vertical projections (columns in rect col range, rows outside rect)
    for c in range(cmin, cmax + 1):
        # Above rect
        above = [(r, grid[r][c]) for r in range(rmin) if grid[r][c] != bg]
        if above:
            color = above[0][1]
            top_r = min(r for r, _ in above)
            for r in range(top_r, rmin):
                out[r][c] = color

        # Below rect
        below = [(r, grid[r][c]) for r in range(rmax + 1, H) if grid[r][c] != bg]
        if below:
            color = below[0][1]
            bot_r = max(r for r, _ in below)
            for r in range(rmax + 1, bot_r + 1):
                out[r][c] = color

    # Horizontal projections (rows in rect row range, cols outside rect)
    for r in range(rmin, rmax + 1):
        # Right of rect
        right = [(c2, grid[r][c2]) for c2 in range(cmax + 1, W) if grid[r][c2] != bg]
        if right:
            color = right[0][1]
            right_c = max(c2 for c2, _ in right)
            for c2 in range(cmax + 1, right_c + 1):
                out[r][c2] = color

        # Left of rect
        left = [(c2, grid[r][c2]) for c2 in range(cmin) if grid[r][c2] != bg]
        if left:
            color = left[0][1]
            left_c = min(c2 for c2, _ in left)
            for c2 in range(left_c, cmin):
                out[r][c2] = color

    return out


def _find_solid_rect(grid, H, W, bg):
    """Find the largest connected component that forms a solid rectangle."""
    visited = [[False] * W for _ in range(H)]
    best_cc = []

    for sr in range(H):
        for sc in range(W):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            color = grid[sr][sc]
            cc = []
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= H or c < 0 or c >= W:
                    continue
                if visited[r][c] or grid[r][c] != color:
                    continue
                visited[r][c] = True
                cc.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))
            if len(cc) > len(best_cc):
                best_cc = cc

    if len(best_cc) >= 4:
        rmin = min(r for r, c in best_cc)
        rmax = max(r for r, c in best_cc)
        cmin = min(c for r, c in best_cc)
        cmax = max(c for r, c in best_cc)
        expected = (rmax - rmin + 1) * (cmax - cmin + 1)
        if len(best_cc) == expected:
            return (rmin, rmax, cmin, cmax)

    return None


def _find_rect_from_pattern(grid, H, W, bg):
    """Fallback: infer invisible rect from single-color column/row runs."""
    # Columns where all non-bg pixels share one color
    single_cols = set()
    for c in range(W):
        colors = set()
        for r in range(H):
            if grid[r][c] != bg:
                colors.add(grid[r][c])
        if len(colors) <= 1:
            single_cols.add(c)

    # Rows where all non-bg pixels share one color
    single_rows = set()
    for r in range(H):
        colors = set()
        for c in range(W):
            if grid[r][c] != bg:
                colors.add(grid[r][c])
        if len(colors) <= 1:
            single_rows.add(r)

    col_run = _longest_contiguous_run(single_cols, W)
    row_run = _longest_contiguous_run(single_rows, H)

    if col_run and row_run:
        return (min(row_run), max(row_run), min(col_run), max(col_run))
    return None


def _longest_contiguous_run(items, total):
    best = []
    cur = []
    for i in range(total):
        if i in items:
            cur.append(i)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best if best else None
