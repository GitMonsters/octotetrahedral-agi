def solve(grid):
    """Find the 1-shape, split at vertical midpoint, change bottom half to 2."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find row extent of 1s
    one_rows = [r for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not one_rows:
        return out

    min_r = min(one_rows)
    max_r = max(one_rows)
    midpoint = (min_r + max_r) / 2.0

    # Change 1s in bottom half to 2
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and r > midpoint:
                out[r][c] = 2

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
