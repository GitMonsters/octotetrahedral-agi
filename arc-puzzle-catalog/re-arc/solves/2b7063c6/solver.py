def transform(input_grid):
    from collections import Counter

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find largest solid rectangle of non-bg color
    best_area = 0
    best_rect = None

    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] == bg:
                continue
            color = grid[r1][c1]
            max_c2 = cols
            for r2 in range(r1, rows):
                if grid[r2][c1] != color:
                    break
                c = c1
                while c < max_c2 and grid[r2][c] == color:
                    c += 1
                max_c2 = c
                area = (r2 - r1 + 1) * (max_c2 - c1)
                if area > best_area:
                    best_area = area
                    best_rect = (r1, r2 + 1, c1, max_c2)

    r1, r2, c1, c2 = best_rect  # r2, c2 exclusive

    output = [row[:] for row in grid]

    # Left side: for each row in rectangle range, look left
    for r in range(r1, r2):
        non_bg = [c for c in range(c1) if grid[r][c] != bg]
        if non_bg:
            color = grid[r][non_bg[0]]
            for c in range(non_bg[0], c1):
                output[r][c] = color

    # Right side: for each row in rectangle range, look right
    for r in range(r1, r2):
        non_bg = [c for c in range(c2, cols) if grid[r][c] != bg]
        if non_bg:
            color = grid[r][non_bg[0]]
            for c in range(c2, non_bg[-1] + 1):
                output[r][c] = color

    # Top side: for each col in rectangle range, look up
    for c in range(c1, c2):
        non_bg = [r for r in range(r1) if grid[r][c] != bg]
        if non_bg:
            color = grid[non_bg[0]][c]
            for r in range(non_bg[0], r1):
                output[r][c] = color

    # Bottom side: for each col in rectangle range, look down
    for c in range(c1, c2):
        non_bg = [r for r in range(r2, rows) if grid[r][c] != bg]
        if non_bg:
            color = grid[non_bg[0]][c]
            for r in range(r2, non_bg[-1] + 1):
                output[r][c] = color

    return output
