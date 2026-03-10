def solve(grid):
    """Scale each cell to a 2x2 block."""
    out = []
    for row in grid:
        new_row = []
        for v in row:
            new_row.extend([v, v])
        out.append(new_row)
        out.append(new_row[:])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
