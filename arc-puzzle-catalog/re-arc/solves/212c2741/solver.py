def transform(grid):
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    freq = Counter()
    for r in range(rows):
        for c in range(cols):
            freq[grid[r][c]] += 1
    colors = freq.most_common()
    bg = colors[0][0]

    other_colors = [co for co, _ in colors[1:]]

    def is_rectangle(color):
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if not positions:
            return False, None
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)
        expected = (max_r - min_r + 1) * (max_c - min_c + 1)
        return len(positions) == expected, (min_r, max_r, min_c, max_c)

    rect_color = None
    scatter_color = None
    rect_bounds = None
    for color in other_colors:
        is_rect, bounds = is_rectangle(color)
        if is_rect:
            rect_color = color
            rect_bounds = bounds
        else:
            scatter_color = color

    if rect_bounds is None:
        return [row[:] for row in grid]
    
    top_r, bot_r, left_c, right_c = rect_bounds

    scatter_pts = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == scatter_color]

    projected = []
    for r, c in scatter_pts:
        above = r < top_r
        below = r > bot_r
        left = c < left_c
        right = c > right_c
        in_row = top_r <= r <= bot_r
        in_col = left_c <= c <= right_c

        if above and in_col:
            projected.append((top_r - 1, c))
        elif below and in_col:
            projected.append((bot_r + 1, c))
        elif left and in_row:
            projected.append((r, left_c - 1))
        elif right and in_row:
            projected.append((r, right_c + 1))
        elif above and left:
            projected.append((top_r - 1, left_c - 1))
        elif above and right:
            projected.append((top_r - 1, right_c + 1))
        elif below and left:
            projected.append((bot_r + 1, left_c - 1))
        elif below and right:
            projected.append((bot_r + 1, right_c + 1))

    out = [[bg] * cols for _ in range(rows)]
    for r in range(top_r, bot_r + 1):
        for c in range(left_c, right_c + 1):
            out[r][c] = rect_color
    for r, c in projected:
        out[r][c] = scatter_color

    return out
