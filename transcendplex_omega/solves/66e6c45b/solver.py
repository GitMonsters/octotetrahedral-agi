def solve(grid):
    """Move center 2x2 values to the four corners of a 4x4 grid."""
    rows = len(grid)
    cols = len(grid[0])
    out = [[0] * cols for _ in range(rows)]
    # Center 2x2 is at rows 1-2, cols 1-2
    out[0][0] = grid[1][1]
    out[0][cols - 1] = grid[1][2]
    out[rows - 1][0] = grid[2][1]
    out[rows - 1][cols - 1] = grid[2][2]
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
