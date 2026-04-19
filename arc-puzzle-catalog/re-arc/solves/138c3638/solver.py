def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Determine background (most common color)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-background cells
    non_bg = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                non_bg.append((r, c))

    if not non_bg:
        return [row[:] for row in input_grid]

    # Bounding box of non-background cells
    min_r = min(r for r, c in non_bg)
    max_r = max(r for r, c in non_bg)
    min_c = min(c for r, c in non_bg)
    max_c = max(c for r, c in non_bg)

    # Fill color: 4 in training examples; if bg is 4, use 0
    fill = 4 if bg != 4 else 0

    output = [row[:] for row in input_grid]

    # Fill bounding box interior (bg cells become fill; frame cells stay)
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if output[r][c] == bg:
                output[r][c] = fill

    # Extend lines outward from gaps in edges
    # Top edge gaps → extend upward
    for c in range(min_c, max_c + 1):
        if input_grid[min_r][c] == bg:
            for r2 in range(0, min_r):
                output[r2][c] = fill

    # Bottom edge gaps → extend downward
    for c in range(min_c, max_c + 1):
        if input_grid[max_r][c] == bg:
            for r2 in range(max_r + 1, rows):
                output[r2][c] = fill

    # Left edge gaps → extend leftward
    for r in range(min_r, max_r + 1):
        if input_grid[r][min_c] == bg:
            for c2 in range(0, min_c):
                output[r][c2] = fill

    # Right edge gaps → extend rightward
    for r in range(min_r, max_r + 1):
        if input_grid[r][max_c] == bg:
            for c2 in range(max_c + 1, cols):
                output[r][c2] = fill

    return output
