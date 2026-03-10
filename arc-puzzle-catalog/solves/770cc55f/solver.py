def solve(grid):
    # Grid has colored row at top, row of 2s in middle, colored row at bottom.
    # Intersection of colored columns from top and bottom rows gets filled with 4s.
    # Fill happens on the side (top or bottom half) with MORE colored cells.
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find the divider row (all 2s)
    div_row = None
    for r in range(rows):
        if all(grid[r][c] == 2 for c in range(cols)):
            div_row = r
            break

    # First row colored columns
    first_cols = set(c for c in range(cols) if grid[0][c] != 0)
    # Last row colored columns
    last_cols = set(c for c in range(cols) if grid[rows-1][c] != 0)

    # Intersection
    intersection = first_cols & last_cols

    # Determine which side to fill
    if len(first_cols) >= len(last_cols):
        # Fill top half (between first row and divider)
        for r in range(1, div_row):
            for c in intersection:
                result[r][c] = 4
    else:
        # Fill bottom half (between divider and last row)
        for r in range(div_row + 1, rows - 1):
            for c in intersection:
                result[r][c] = 4

    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
