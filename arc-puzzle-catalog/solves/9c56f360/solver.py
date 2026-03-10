def solve(grid):
    """Each row's contiguous block of 3s slides left until hitting an 8 or the edge."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    for r in range(rows):
        # Find contiguous 3s in this row
        threes = [c for c in range(cols) if grid[r][c] == 3]
        if not threes:
            continue
        min_c = min(threes)
        count = len(threes)

        # Clear original 3 positions
        for c in threes:
            out[r][c] = 0

        # Find leftmost position: slide left from min_c until hitting 8 or edge
        # The block occupies [new_start, new_start+count-1]
        new_start = min_c
        while new_start > 0 and out[r][new_start - 1] != 8:
            new_start -= 1

        for i in range(count):
            out[r][new_start + i] = 3

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
