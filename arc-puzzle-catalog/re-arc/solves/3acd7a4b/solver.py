def find_largest_rect(mask):
    """Returns (area, r1, c1, r2, c2) of largest all-1s rectangle."""
    if not mask or not mask[0]:
        return 0, 0, 0, 0, 0
    R, C = len(mask), len(mask[0])
    heights = [0] * C
    best = (0, 0, 0, 0, 0)
    for r in range(R):
        for c in range(C):
            heights[c] = heights[c] + 1 if mask[r][c] else 0
        stack = []
        for c in range(C + 1):
            h = heights[c] if c < C else 0
            while stack and heights[stack[-1]] > h:
                idx = stack.pop()
                height = heights[idx]
                left = stack[-1] + 1 if stack else 0
                width = c - left
                area = height * width
                if area > best[0]:
                    best = (area, r - height + 1, left, r, c - 1)
            stack.append(c)
    return best


def transform(grid: list[list[int]]) -> list[list[int]]:
    R, C = len(grid), len(grid[0])
    colors = sorted(set(v for row in grid for v in row))

    # Find the largest rectangle containing exactly 2 colors
    best_area = 0
    best_rect = None
    best_colors = None
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            c1, c2 = colors[i], colors[j]
            mask = [[1 if grid[r][c] in (c1, c2) else 0 for c in range(C)] for r in range(R)]
            area, r1, cl, r2, cr = find_largest_rect(mask)
            if area > best_area:
                best_area = area
                best_rect = (r1, cl, r2, cr)
                best_colors = (c1, c2)

    r1, c1, r2, c2 = best_rect
    ca, cb = best_colors

    # Dominant color fills most of the rectangle; minority is the sparse dots
    count_a = sum(1 for r in range(r1, r2 + 1) for c in range(c1, c2 + 1) if grid[r][c] == ca)
    count_b = sum(1 for r in range(r1, r2 + 1) for c in range(c1, c2 + 1) if grid[r][c] == cb)
    D, M = (ca, cb) if count_a >= count_b else (cb, ca)

    # Collect which rows/cols contain a minority-color dot
    dot_rows = set()
    dot_cols = set()
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if grid[r][c] == M:
                dot_rows.add(r - r1)
                dot_cols.add(c - c1)

    # Output: M where row OR col has a dot, D otherwise
    H = r2 - r1 + 1
    W = c2 - c1 + 1
    return [
        [M if r in dot_rows or c in dot_cols else D for c in range(W)]
        for r in range(H)
    ]
