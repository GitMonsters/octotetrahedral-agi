def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find separator rows (all cells same value) to determine grid_color
    sep_rows = []
    grid_color = None
    for r in range(rows):
        if len(set(input_grid[r])) == 1:
            sep_rows.append(r)
            grid_color = input_grid[r][0]

    # Find separator columns (all cells equal grid_color)
    sep_cols = []
    for c in range(cols):
        if all(input_grid[r][c] == grid_color for r in range(rows)):
            sep_cols.append(c)

    # Determine row bands (content rows between separators)
    row_bands = []
    boundaries = [-1] + sorted(sep_rows) + [rows]
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1] - 1
        if start <= end:
            row_bands.append((start, end))

    # Determine col bands
    col_bands = []
    boundaries = [-1] + sorted(sep_cols) + [cols]
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1] - 1
        if start <= end:
            col_bands.append((start, end))

    # Find background color (most common non-grid color in blocks)
    color_count = {}
    for r_start, r_end in row_bands:
        for c_start, c_end in col_bands:
            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    v = input_grid[r][c]
                    if v != grid_color:
                        color_count[v] = color_count.get(v, 0) + 1
    bg_color = max(color_count, key=color_count.get)

    # Create output grid
    output = [row[:] for row in input_grid]

    # Process each block
    for r_start, r_end in row_bands:
        for c_start, c_end in col_bands:
            # Find the single marker (first non-bg cell in block)
            marker = None
            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    if input_grid[r][c] != bg_color:
                        marker = input_grid[r][c]
                        break
                if marker is not None:
                    break

            # Determine fill color based on marker
            if marker is None or marker == 3:
                fill = bg_color
            elif marker == 5 or marker == 6:
                fill = 7
            elif marker == grid_color:
                fill = 3
            else:
                fill = 7

            # Fill the block uniformly
            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    output[r][c] = fill

    return output
