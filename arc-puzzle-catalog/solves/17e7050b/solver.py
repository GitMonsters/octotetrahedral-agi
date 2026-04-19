def transform(grid):
    from collections import Counter
    rows, cols = len(grid), len(grid[0])
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    out = [row[:] for row in grid]

    best_rect = None
    best_area = 0
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] == bg:
                continue
            color = grid[r1][c1]
            c2 = c1
            while c2 + 1 < cols and grid[r1][c2+1] == color:
                c2 += 1
            for r2 in range(r1, rows):
                if all(grid[r2][c] == color for c in range(c1, c2+1)):
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, c1, r2, c2)
                else:
                    break
    if best_rect is None:
        return out

    r1, c1, r2, c2 = best_rect

    for c in range(c1, c2+1):
        dots = [(r, grid[r][c]) for r in range(r1) if grid[r][c] != bg]
        if dots:
            fr = min(r for r, _ in dots)
            col = dots[0][1]
            for r in range(fr, r1):
                out[r][c] = col
        dots = [(r, grid[r][c]) for r in range(r2+1, rows) if grid[r][c] != bg]
        if dots:
            fr = max(r for r, _ in dots)
            col = dots[0][1]
            for r in range(r2+1, fr+1):
                out[r][c] = col

    for r in range(r1, r2+1):
        dots = [(c, grid[r][c]) for c in range(c1) if grid[r][c] != bg]
        if dots:
            fc = min(c for c, _ in dots)
            col = dots[0][1]
            for c in range(fc, c1):
                out[r][c] = col
        dots = [(c, grid[r][c]) for c in range(c2+1, cols) if grid[r][c] != bg]
        if dots:
            fc = max(c for c, _ in dots)
            col = dots[0][1]
            for c in range(c2+1, fc+1):
                out[r][c] = col

    return out
