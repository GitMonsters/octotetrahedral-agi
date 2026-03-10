def solve(grid: list[list[int]]) -> list[list[int]]:
    """XOR left and right halves separated by column of 4s. Output 2 where exactly one is non-zero."""
    rows = len(grid)
    cols = len(grid[0])

    div = next(c for c in range(cols) if all(grid[r][c] == 4 for r in range(rows)))
    w = div

    out = []
    for r in range(rows):
        row = []
        for c in range(w):
            left = grid[r][c] != 0
            right = grid[r][div + 1 + c] != 0
            row.append(2 if left != right else 0)
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
