def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find uniform rows and columns (candidates for cross separators)
    uniform_rows = {}
    for r in range(rows):
        if len(set(input_grid[r])) == 1:
            color = input_grid[r][0]
            uniform_rows.setdefault(color, []).append(r)

    uniform_cols = {}
    for c in range(cols):
        col_vals = set(input_grid[r][c] for r in range(rows))
        if len(col_vals) == 1:
            color = input_grid[0][c]
            uniform_cols.setdefault(color, []).append(c)

    # Find the separator: matching color between a uniform row and column
    sep_row = sep_col = sep_color = None
    for color in uniform_rows:
        if color in uniform_cols:
            sep_color = color
            sep_row = uniform_rows[color][0]
            sep_col = uniform_cols[color][0]
            break

    # Define 4 regions (excluding separator row/col)
    region_bounds = {
        'TL': (0, sep_row, 0, sep_col),
        'TR': (0, sep_row, sep_col + 1, cols),
        'BL': (sep_row + 1, rows, 0, sep_col),
        'BR': (sep_row + 1, rows, sep_col + 1, cols),
    }

    def extract(r0, r1, c0, c1):
        return [[input_grid[r][c] for c in range(c0, c1)] for r in range(r0, r1)]

    def region_size(b):
        return (b[1] - b[0]) * (b[3] - b[2])

    # Content = largest region, indicator = diagonally opposite
    sizes = {n: region_size(b) for n, b in region_bounds.items()}
    content_name = max(sizes, key=sizes.get)
    diagonal = {'TL': 'BR', 'TR': 'BL', 'BL': 'TR', 'BR': 'TL'}
    indicator_name = diagonal[content_name]
    border_names = [n for n in region_bounds if n not in (content_name, indicator_name)]

    content = extract(*region_bounds[content_name])
    indicator = extract(*region_bounds[indicator_name])

    content_rows = len(content)
    content_cols = len(content[0])
    ind_rows = len(indicator)
    ind_cols = len(indicator[0])

    # Determine the two colors in content (bg = most common, noise = other)
    from collections import Counter
    color_counts = Counter()
    for row in content:
        color_counts.update(row)
    colors = color_counts.most_common()
    bg_color = colors[0][0]
    noise_color = colors[1][0]

    # Border fill color determines which content color stays vs gets replaced
    border_data = extract(*region_bounds[border_names[0]])
    border_color = border_data[0][0]

    if border_color == bg_color:
        replace_color = noise_color  # noise gets replaced, bg stays
    else:
        replace_color = bg_color  # bg gets replaced, noise stays

    # Sub-quadrant dimensions
    sub_h = content_rows // ind_rows
    sub_w = content_cols // ind_cols

    # Build output: replace_color -> indicator[si][sj], other color stays
    output = []
    for r in range(content_rows):
        row = []
        for c in range(content_cols):
            si = min(r // sub_h, ind_rows - 1)
            sj = min(c // sub_w, ind_cols - 1)
            if content[r][c] == replace_color:
                row.append(indicator[si][sj])
            else:
                row.append(content[r][c])
        output.append(row)

    return output
