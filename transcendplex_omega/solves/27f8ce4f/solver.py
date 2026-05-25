def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find most common color. Place input copies at positions matching that color in the 3x3 grid."""
    from collections import Counter
    n = len(grid)
    counts: Counter = Counter()
    for r in range(n):
        for c in range(n):
            counts[grid[r][c]] += 1
    most_common = counts.most_common(1)[0][0]

    out = [[0]*(n*n) for _ in range(n*n)]
    for br in range(n):
        for bc in range(n):
            if grid[br][bc] == most_common:
                for r in range(n):
                    for c in range(n):
                        out[br*n + r][bc*n + c] = grid[r][c]

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
