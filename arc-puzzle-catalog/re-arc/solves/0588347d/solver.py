from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find cross lines: row/col where all values are the same non-bg color
    cross_row = cross_col = cross_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != bg:
            cross_row = r
            cross_color = grid[r][0]
    for c in range(cols):
        if len(set(grid[r][c] for r in range(rows))) == 1 and grid[0][c] != bg:
            cross_col = c

    # Find marker pixels (non-bg, not on cross lines)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and r != cross_row and c != cross_col:
                markers.append((r, c))

    n = len(markers)

    # Markers in one quadrant → move cross opposite direction by n
    if n > 0:
        above = all(r < cross_row for r, c in markers)
        left = all(c < cross_col for r, c in markers)
        delta_r = n if above else -n
        delta_c = n if left else -n
    else:
        delta_r = delta_c = 0

    new_row = cross_row + delta_r
    new_col = cross_col + delta_c

    # Build output: background + new cross
    output = [[bg] * cols for _ in range(rows)]
    for c in range(cols):
        output[new_row][c] = cross_color
    for r in range(rows):
        output[r][new_col] = cross_color

    return output
