def solve(grid):
    """Fractal self-similar tiling: 4x4 → 16x16. Each non-zero cell → input pattern, zero → all zeros."""
    N = len(grid)
    out = [[0] * (N * N) for _ in range(N * N)]
    for br in range(N):
        for bc in range(N):
            if grid[br][bc] != 0:
                for r in range(N):
                    for c in range(N):
                        out[br * N + r][bc * N + c] = grid[r][c]
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
