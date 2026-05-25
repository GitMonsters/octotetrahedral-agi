def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # For each cell with value 1, fill 0s in 3x3 neighborhood with 1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                            out[nr][nc] = 1

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
