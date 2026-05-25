def solve(grid):
    """Tile input 2x2: TL=rot180, TR=flipud, BL=fliplr, BR=identity."""
    n = len(grid)
    m = len(grid[0])

    def rot180(g):
        return [[g[len(g) - 1 - i][len(g[0]) - 1 - j] for j in range(len(g[0]))] for i in range(len(g))]

    def flipud(g):
        return [row[:] for row in reversed(g)]

    def fliplr(g):
        return [row[::-1] for row in g]

    tl = rot180(grid)
    tr = flipud(grid)
    bl = fliplr(grid)
    br = [row[:] for row in grid]

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
