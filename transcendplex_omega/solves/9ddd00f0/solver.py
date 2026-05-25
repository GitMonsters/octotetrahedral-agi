def solve(grid):
    """Grid is NxN where N = K²+K-1. Divided into KxK blocks of size KxK.
    Each block position (br,bc) gets filled with 'color' except a hole at (br,bc)."""
    n = len(grid)

    # Find K: n = K*(K+1) - 1
    k = None
    for candidate in range(1, n + 1):
        if candidate * (candidate + 1) - 1 == n:
            k = candidate
            break

    # Find the non-zero color
    color = 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0:
                color = grid[r][c]
                break
        if color != 0:
            break

    # Build output
    result = [[0] * n for _ in range(n)]
    for br in range(k):
        for bc in range(k):
            rs = br * (k + 1)
            cs = bc * (k + 1)
            for lr in range(k):
                for lc in range(k):
                    if lr == br and lc == bc:
                        result[rs + lr][cs + lc] = 0
                    else:
                        result[rs + lr][cs + lc] = color
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
