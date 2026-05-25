def solve(grid):
    """Tile the 3x3 input NxN times where N = number of distinct colors."""
    colors = set()
    for row in grid:
        for cell in row:
            colors.add(cell)
    n = len(colors)
    rows = len(grid)
    cols = len(grid[0])
    result = []
    for r in range(rows * n):
        result.append([grid[r % rows][c % cols] for c in range(cols * n)])
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
