def transform(input_grid):
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    output = copy.deepcopy(input_grid)

    # Find background color (most common)
    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[input_grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]

    # Find the dot (single non-background pixel)
    dot_row, dot_col, dot_color = None, None, None
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                dot_row, dot_col, dot_color = r, c, input_grid[r][c]
                break
        if dot_row is not None:
            break

    if dot_row is None:
        return output

    dot_parity = dot_col % 2

    # Fill affected zone: cols 0..dot_col, all rows
    # Columns with same parity as dot_col → dot_color
    for r in range(rows):
        for c in range(dot_col + 1):
            if c % 2 == dot_parity:
                output[r][c] = dot_color

    # Place maroon (9) on non-dot-color columns at boundary rows
    # Enumerate non-dot cols from nearest to dot going left
    non_dot_cols = []
    for c in range(dot_col - 1, -1, -1):
        if c % 2 != dot_parity:
            non_dot_cols.append(c)

    for i, c in enumerate(non_dot_cols):
        if i % 2 == 0:
            output[0][c] = 9        # odd-indexed from dot → top row
        else:
            output[dot_row][c] = 9   # even-indexed → dot's row

    return output
