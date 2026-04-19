def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most frequent)
    all_colors = [c for row in input_grid for c in row]
    bg = Counter(all_colors).most_common(1)[0][0]

    # Find separator rows (entirely background color)
    sep_rows = set()
    for r in range(rows):
        if all(input_grid[r][c] == bg for c in range(cols)):
            sep_rows.add(r)

    # Find separator columns (entirely background color)
    sep_cols = set()
    for c in range(cols):
        if all(input_grid[r][c] == bg for r in range(rows)):
            sep_cols.add(c)

    # Extract contiguous row sections
    row_sections = []
    current = []
    for r in range(rows):
        if r in sep_rows:
            if current:
                row_sections.append(current)
                current = []
        else:
            current.append(r)
    if current:
        row_sections.append(current)

    # Extract contiguous column sections
    col_sections = []
    current = []
    for c in range(cols):
        if c in sep_cols:
            if current:
                col_sections.append(current)
                current = []
        else:
            current.append(c)
    if current:
        col_sections.append(current)

    # Build output: most common non-background color in each block
    output = []
    for rs in row_sections:
        row = []
        for cs in col_sections:
            colors = []
            for r in rs:
                for c in cs:
                    colors.append(input_grid[r][c])
            non_bg = [c for c in colors if c != bg]
            if non_bg:
                dominant = Counter(non_bg).most_common(1)[0][0]
            else:
                dominant = bg
            row.append(dominant)
        output.append(row)

    return output
