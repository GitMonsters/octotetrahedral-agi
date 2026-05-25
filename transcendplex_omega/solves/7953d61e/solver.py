def solve(grid):
    """Tile input in 2x2: TL=id, TR=rot90CCW, BL=rot180, BR=rot270CCW."""
    n = len(grid)
    m = len(grid[0])

    def rot90ccw(g):
        # new[i][j] = g[j][n-1-i] where n=rows of g
        r, c = len(g), len(g[0])
        return [[g[j][r - 1 - i] for j in range(c)] for i in range(r)]

    def rot180(g):
        r, c = len(g), len(g[0])
        return [[g[r - 1 - i][c - 1 - j] for j in range(c)] for i in range(r)]

    def rot270ccw(g):
        # = rot90CW: new[i][j] = g[n-1-j][i]
        r, c = len(g), len(g[0])
        return [[g[c - 1 - j][i] for j in range(c)] for i in range(r)]

    tl = [row[:] for row in grid]
    tr = rot90ccw(grid)
    bl = rot180(grid)
    br = rot270ccw(grid)

    out = []
    for i in range(n):
        out.append(tl[i] + tr[i])
    for i in range(n):
        out.append(bl[i] + br[i])
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
