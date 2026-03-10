def solve(grid):
    """OR two halves split by a row of 4s: non-zero in either half → 3."""
    R, C = len(grid), len(grid[0])
    # Find the separator row (all 4s)
    sep = None
    for r in range(R):
        if all(v == 4 for v in grid[r]):
            sep = r
            break
    top = grid[:sep]
    bot = grid[sep + 1:]
    out = []
    for r in range(len(top)):
        row = []
        for c in range(C):
            if top[r][c] != 0 or bot[r][c] != 0:
                row.append(3)
            else:
                row.append(0)
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
