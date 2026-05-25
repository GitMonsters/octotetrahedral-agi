def solve(grid):
    """Fill interior of V/triangle shapes (between colored arms) with color 2."""
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    for r in range(rows):
        colored_cols = [c for c in range(cols) if grid[r][c] != 0]
        if len(colored_cols) >= 2:
            lo, hi = min(colored_cols), max(colored_cols)
            for c in range(lo + 1, hi):
                if result[r][c] == 0:
                    result[r][c] = 2
    return result


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
