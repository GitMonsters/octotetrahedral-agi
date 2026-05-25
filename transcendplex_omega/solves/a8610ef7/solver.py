def solve(grid):
    # Rule: for each 8-cell, if the vertically flipped cell is also 8 -> 2, else -> 5
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                flipped_r = rows - 1 - r
                if grid[flipped_r][c] == 8:
                    result[r][c] = 2
                else:
                    result[r][c] = 5
    return result

if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
