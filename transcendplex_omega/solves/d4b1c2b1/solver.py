def solve(grid):
    distinct = len(set(v for row in grid for v in row))
    scale = distinct
    rows = len(grid)
    cols = len(grid[0])
    out = []
    for r in range(rows):
        for _ in range(scale):
            row = []
            for c in range(cols):
                row.extend([grid[r][c]] * scale)
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
