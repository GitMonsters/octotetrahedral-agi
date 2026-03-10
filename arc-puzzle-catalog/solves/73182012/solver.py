def solve(grid):
    """Find bounding box of non-zero cells, extract top-left quadrant."""
    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r = rows, 0
    min_c, max_c = cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    h = (max_r - min_r + 1) // 2
    w = (max_c - min_c + 1) // 2
    out = []
    for r in range(min_r, min_r + h):
        out.append([grid[r][c] for c in range(min_c, min_c + w)])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
