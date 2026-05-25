def solve(grid):
    """Group consecutive identical rows into blocks. Every 3rd block (1st, 4th, 7th...)
    has its non-zero values replaced with 2."""
    rows = len(grid)
    out = [row[:] for row in grid]

    # Group consecutive identical rows into blocks
    blocks = []  # list of (start_row, end_row) inclusive
    i = 0
    while i < rows:
        j = i + 1
        while j < rows and grid[j] == grid[i]:
            j += 1
        blocks.append((i, j - 1))
        i = j

    # Every block at index 0, 3, 6, 9... gets changed
    for idx, (start, end) in enumerate(blocks):
        if idx % 3 == 0:
            for r in range(start, end + 1):
                for c in range(len(grid[r])):
                    if out[r][c] != 0:
                        out[r][c] = 2

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
