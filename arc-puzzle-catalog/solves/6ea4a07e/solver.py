def solve(grid):
    # Rule: invert 0/non-zero pattern, map color: 8->2, 3->1, 5->4
    color_map = {8: 2, 3: 1, 5: 4}
    rows, cols = len(grid), len(grid[0])
    # Find the non-zero color in the grid
    color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                break
        if color is not None:
            break
    new_color = color_map[color]
    return [[new_color if grid[r][c] == 0 else 0 for c in range(cols)] for r in range(rows)]

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
