def solve(grid):
    n = len(grid)
    m = len(grid[0])

    def rot90cw(g):
        """Rotate 90 degrees clockwise."""
        r = len(g)
        c = len(g[0])
        return [[g[r - 1 - j][i] for j in range(r)] for i in range(c)]

    def rot90ccw(g):
        """Rotate 90 degrees counter-clockwise."""
        r = len(g)
        c = len(g[0])
        return [[g[j][c - 1 - i] for j in range(r)] for i in range(c)]

    def rot180(g):
        """Rotate 180 degrees."""
        r = len(g)
        c = len(g[0])
        return [[g[r - 1 - i][c - 1 - j] for j in range(c)] for i in range(r)]

    identity = grid
    top_right = rot90ccw(grid)
    bottom_left = rot180(grid)
    bottom_right = rot90cw(grid)

    out = []
    # Top half
    for r in range(n):
        out.append(identity[r] + top_right[r])
    # Bottom half
    for r in range(n):
        out.append(bottom_left[r] + bottom_right[r])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
