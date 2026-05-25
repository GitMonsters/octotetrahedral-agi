def solve(grid):
    """Scale down by factor 2: each 2x2 block maps to its non-zero value."""
    rows = len(grid)
    cols = len(grid[0])
    out_rows = rows // 2
    out_cols = cols // 2
    out = [[0] * out_cols for _ in range(out_rows)]
    for r in range(out_rows):
        for c in range(out_cols):
            block = [
                grid[2 * r][2 * c], grid[2 * r][2 * c + 1],
                grid[2 * r + 1][2 * c], grid[2 * r + 1][2 * c + 1],
            ]
            for v in block:
                if v != 0:
                    out[r][c] = v
                    break
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
