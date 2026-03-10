def solve(grid):
    """Left pattern + fill with border color + duplicate left pattern on right."""
    R, C = len(grid), len(grid[0])
    # Find pattern width: first zero in first row
    pw = C
    for c in range(C):
        if grid[0][c] == 0:
            pw = c
            break
    out = []
    for r in range(R):
        left = grid[r][:pw]
        border = left[0]
        fill_len = C - 2 * pw
        out.append(left + [border] * fill_len + left)
    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
