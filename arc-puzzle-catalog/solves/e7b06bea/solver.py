def solve(grid):
    """Collapse stripe columns into a single column, cycling colors with period n (number of 5-rows).
    
    The stripe columns (rightmost non-zero constant columns) merge into one column
    one position to their left, with colors cycling every n rows.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Count 5-rows (rows where col 0 is 5)
    n = 0
    for r in range(rows):
        if grid[r][0] == 5:
            n += 1
        else:
            break

    # Find stripe columns: rightmost columns with constant non-zero, non-5 values
    stripe_colors = []
    stripe_start_col = None
    for c in range(cols):
        val = grid[0][c]
        if val != 0 and val != 5 and all(grid[r][c] == val for r in range(rows)):
            if stripe_start_col is None:
                stripe_start_col = c
            stripe_colors.append(val)

    output_col = stripe_start_col - 1

    # Build output
    out = [[0] * cols for _ in range(rows)]
    # Preserve 5s
    for r in range(n):
        out[r][0] = 5

    # Fill the output column with cycling colors
    color_idx = 0
    row_in_group = 0
    for r in range(rows):
        out[r][output_col] = stripe_colors[color_idx % len(stripe_colors)]
        row_in_group += 1
        if row_in_group >= n:
            row_in_group = 0
            color_idx += 1

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
