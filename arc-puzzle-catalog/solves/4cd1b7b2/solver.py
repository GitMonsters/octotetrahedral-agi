def solve(grid):
    """Complete a 4x4 Latin square: fill 0s so each row/col has {1,2,3,4}."""
    g = [row[:] for row in grid]
    full = {1, 2, 3, 4}

    def backtrack():
        for r in range(4):
            for c in range(4):
                if g[r][c] == 0:
                    row_vals = {g[r][cc] for cc in range(4) if g[r][cc] != 0}
                    col_vals = {g[rr][c] for rr in range(4) if g[rr][c] != 0}
                    for v in full - row_vals - col_vals:
                        g[r][c] = v
                        if backtrack():
                            return True
                        g[r][c] = 0
                    return False
        return True

    backtrack()
    return g


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
