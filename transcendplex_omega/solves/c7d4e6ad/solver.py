def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(rows):
        color = grid[r][0]
        if color == 0:
            continue
        for c in range(cols):
            if grid[r][c] == 5:
                out[r][c] = color
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
