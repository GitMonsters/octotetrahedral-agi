def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Background = most common color
    flat = [input_grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    colors = set(flat) - {bg}

    best_area = 0
    best_rect = None  # (r1, c1, r2, c2, color)

    for color in colors:
        # For each cell, width of consecutive `color` cells to the right
        w = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols - 1, -1, -1):
                if input_grid[r][c] == color:
                    w[r][c] = (w[r][c + 1] if c + 1 < cols else 0) + 1
                else:
                    w[r][c] = 0

        # For each starting cell, sweep downward tracking min width
        for r1 in range(rows):
            for c1 in range(cols):
                if w[r1][c1] == 0:
                    continue
                min_w = w[r1][c1]
                for r2 in range(r1, rows):
                    if w[r2][c1] == 0:
                        break
                    min_w = min(min_w, w[r2][c1])
                    area = (r2 - r1 + 1) * min_w
                    if area > best_area:
                        best_area = area
                        best_rect = (r1, c1, r2, c1 + min_w - 1, color)

    # Build output
    output = [[bg] * cols for _ in range(rows)]
    if best_rect:
        r1, c1, r2, c2, color = best_rect
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                output[r][c] = color

    return output
