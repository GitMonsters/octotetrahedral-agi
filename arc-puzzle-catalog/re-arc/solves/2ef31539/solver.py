def transform(input_grid):
    """
    Rule: A cross (full row + full column of one color) divides the grid into 4 quadrants.
    One quadrant contains a pattern; the others are empty (background).
    Copy the pattern into all 4 quadrants.
    """
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find divider row (all cells identical and non-background)
    div_row = None
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != bg:
            div_row = r
            break

    # Find divider column
    div_col = None
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] != bg:
            div_col = c
            break

    # Extract quadrant content
    def extract(r1, r2, c1, c2):
        return [[grid[r][c] for c in range(c1, c2 + 1)] for r in range(r1, r2 + 1)]

    def is_empty(quad):
        return all(c == bg for row in quad for c in row)

    quads = [
        (0, div_row - 1, 0, div_col - 1),                    # TL
        (0, div_row - 1, div_col + 1, cols - 1),              # TR
        (div_row + 1, rows - 1, 0, div_col - 1),              # BL
        (div_row + 1, rows - 1, div_col + 1, cols - 1),       # BR
    ]

    # Find the non-empty quadrant (source)
    source = None
    for q in quads:
        data = extract(*q)
        if not is_empty(data):
            source = data
            break

    # Paste source into all quadrants
    for r1, r2, c1, c2 in quads:
        for i, r in enumerate(range(r1, r2 + 1)):
            for j, c in enumerate(range(c1, c2 + 1)):
                grid[r][c] = source[i][j]

    return grid
