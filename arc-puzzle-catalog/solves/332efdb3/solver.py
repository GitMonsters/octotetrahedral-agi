def solve(grid: list[list[int]]) -> list[list[int]]:
    """output[r][c] = 0 if (r%2==1 and c%2==1) else 1."""
    rows = len(grid)
    cols = len(grid[0])
    return [[0 if (r % 2 == 1 and c % 2 == 1) else 1 for c in range(cols)] for r in range(rows)]


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
