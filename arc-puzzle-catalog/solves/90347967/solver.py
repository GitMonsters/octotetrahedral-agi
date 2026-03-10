def solve(grid):
    # Rotate all non-zero cells 180 degrees about the cell with value 5.
    # Original cells become 0, rotated cells take their place.
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find the cell with value 5 (center of rotation)
    cr, cc = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                cr, cc = r, c
                break
        if cr is not None:
            break

    # Rotate each non-zero cell 180 about (cr, cc)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                nr, nc = 2 * cr - r, 2 * cc - c
                if 0 <= nr < rows and 0 <= nc < cols:
                    result[nr][nc] = grid[r][c]

    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
