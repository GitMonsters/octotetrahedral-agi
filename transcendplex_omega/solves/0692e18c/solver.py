def solve(grid: list[list[int]]) -> list[list[int]]:
    """Each non-zero cell maps to a 3x3 block that is the inverted pattern of the input."""
    n = len(grid)
    color = 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0:
                color = grid[r][c]
                break
        if color:
            break

    inverted = [[0]*n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            inverted[r][c] = 0 if grid[r][c] != 0 else color

    out_size = n * n
    out = [[0]*out_size for _ in range(out_size)]

    for br in range(n):
        for bc in range(n):
            if grid[br][bc] != 0:
                for r in range(n):
                    for c in range(n):
                        out[br*n + r][bc*n + c] = inverted[r][c]

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
