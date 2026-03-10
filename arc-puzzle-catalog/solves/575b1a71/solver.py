def solve(grid):
    """Replace 0s on background of 5s: each column with 0s gets color 1-4 (left to right)."""
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Find which columns contain 0s
    cols_with_zeros = sorted({c for r in range(R) for c in range(C) if grid[r][c] == 0})

    # Map columns to colors 1,2,3,4
    col_to_color = {col: i + 1 for i, col in enumerate(cols_with_zeros)}

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0:
                out[r][c] = col_to_color[c]
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
