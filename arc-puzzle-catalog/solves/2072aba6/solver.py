def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each 5-cell maps to 2x2 block [[1,2],[2,1]]. Zero cells map to 2x2 zeros."""
    r = len(grid)
    c = len(grid[0])
    out = [[0]*(c*2) for _ in range(r*2)]

    for i in range(r):
        for j in range(c):
            if grid[i][j] == 5:
                out[i*2][j*2] = 1
                out[i*2][j*2+1] = 2
                out[i*2+1][j*2] = 2
                out[i*2+1][j*2+1] = 1

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
