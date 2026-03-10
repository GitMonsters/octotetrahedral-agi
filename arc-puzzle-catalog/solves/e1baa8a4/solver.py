def solve(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find row boundaries from column 0
    row_bounds = [0]
    for r in range(1, rows):
        if grid[r][0] != grid[r - 1][0]:
            row_bounds.append(r)

    # Find col boundaries from row 0
    col_bounds = [0]
    for c in range(1, cols):
        if grid[0][c] != grid[0][c - 1]:
            col_bounds.append(c)

    out = []
    for rb in row_bounds:
        row = []
        for cb in col_bounds:
            row.append(grid[rb][cb])
        out.append(row)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
