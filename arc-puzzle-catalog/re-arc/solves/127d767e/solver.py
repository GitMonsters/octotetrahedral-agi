def transform(input_grid):
    """
    Rule: A cross divider splits the grid into 4 quadrants.
    - One quadrant (largest) is the pattern with two colors.
    - The diagonally opposite quadrant (2x2) is the palette.
    - The other two quadrants are filled with the "keep" color.
    - In the output, extract the pattern quadrant and replace the non-keep color
      using the 2x2 palette mapped to the pattern's four quadrants (split at midpoint).
    """
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find uniform rows and columns (candidates for divider)
    uniform_rows = {}
    for r in range(rows):
        if len(set(input_grid[r])) == 1:
            uniform_rows[r] = input_grid[r][0]

    uniform_cols = {}
    for c in range(cols):
        col_vals = set(input_grid[r][c] for r in range(rows))
        if len(col_vals) == 1:
            uniform_cols[c] = input_grid[0][c]

    # Find divider row and col sharing same color
    divider_row = divider_col = divider_color = None
    for r, rc in uniform_rows.items():
        for c, cc in uniform_cols.items():
            if rc == cc:
                divider_row, divider_col, divider_color = r, c, rc
                break
        if divider_row is not None:
            break

    # Extract a quadrant by name
    def get_quadrant(q):
        if q == 'TL':
            return [row[:divider_col] for row in input_grid[:divider_row]]
        elif q == 'TR':
            return [row[divider_col + 1:] for row in input_grid[:divider_row]]
        elif q == 'BL':
            return [row[:divider_col] for row in input_grid[divider_row + 1:]]
        elif q == 'BR':
            return [row[divider_col + 1:] for row in input_grid[divider_row + 1:]]

    # Quadrant sizes
    sizes = {
        'TL': divider_row * divider_col,
        'TR': divider_row * (cols - divider_col - 1),
        'BL': (rows - divider_row - 1) * divider_col,
        'BR': (rows - divider_row - 1) * (cols - divider_col - 1),
    }

    # Pattern = largest quadrant, palette = diagonally opposite (2x2)
    pattern_q = max(sizes, key=sizes.get)
    opposite = {'TL': 'BR', 'TR': 'BL', 'BL': 'TR', 'BR': 'TL'}
    palette_q = opposite[pattern_q]

    pattern = get_quadrant(pattern_q)
    palette = get_quadrant(palette_q)

    # Keep color from the two fill quadrants
    fill_quads = [q for q in ['TL', 'TR', 'BL', 'BR'] if q != pattern_q and q != palette_q]
    fill_data = get_quadrant(fill_quads[0])
    keep_color = fill_data[0][0]

    # Build output: keep_color stays, other color replaced by palette quadrant mapping
    p_rows = len(pattern)
    p_cols = len(pattern[0])
    mid_r = p_rows // 2
    mid_c = p_cols // 2

    output = []
    for r in range(p_rows):
        row_out = []
        for c in range(p_cols):
            if pattern[r][c] == keep_color:
                row_out.append(keep_color)
            else:
                qi = 0 if r < mid_r else 1
                qj = 0 if c < mid_c else 1
                row_out.append(palette[qi][qj])
        output.append(row_out)

    return output
