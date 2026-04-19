def transform(grid):
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    ctr = Counter(flat)
    bg = ctr.most_common(1)[0][0]
    rect_colors = [c for c in ctr if c != bg]
    if not rect_colors:
        return out
    rect_color = rect_colors[0]

    # Find connected rectangles
    visited = [[False] * cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == rect_color and not visited[r][c]:
                min_r, max_r, min_c, max_c = r, r, c, c
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == rect_color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                rects.append((min_r, min_c, max_r, max_c))

    # Build masks: border width = max(height, width) // 2
    border_mask = [[False] * cols for _ in range(rows)]
    rect_mask = [[False] * cols for _ in range(rows)]

    for (r1, c1, r2, c2) in rects:
        B = max(r2 - r1 + 1, c2 - c1 + 1) // 2
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                rect_mask[r][c] = True
        for r in range(max(0, r1 - B), min(rows, r2 + B + 1)):
            for c in range(max(0, c1 - B), min(cols, c2 + B + 1)):
                if not (r1 <= r <= r2 and c1 <= c <= c2):
                    border_mask[r][c] = True

    # For each row: find leftmost border of rects with body on this row
    for r in range(rows):
        body_rects = [(r1, c1, r2, c2) for (r1, c1, r2, c2) in rects if r1 <= r <= r2]
        L = cols
        if body_rects:
            for (r1, c1, r2, c2) in body_rects:
                B = max(r2 - r1 + 1, c2 - c1 + 1) // 2
                L = min(L, max(0, c1 - B))

        for c in range(cols):
            if rect_mask[r][c]:
                pass  # rect body keeps its color
            elif border_mask[r][c]:
                out[r][c] = 2
            elif body_rects and c >= L:
                out[r][c] = 5  # shadow

    return out
