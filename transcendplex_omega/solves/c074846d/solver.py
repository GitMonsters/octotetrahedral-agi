def solve(grid):
    """Find 5 and adjacent 2s. Rotate 2s 90 degrees CCW around 5. Original 2s become 3."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find the 5
    five_r, five_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                five_r, five_c = r, c
                break
        if five_r is not None:
            break

    # Find 2s and their direction relative to 5
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    if not twos:
        return out

    count = len(twos)

    # Determine direction: all 2s should be in one direction from 5
    sample = twos[0]
    dr = sample[0] - five_r
    dc = sample[1] - five_c
    if dr != 0:
        dr = dr // abs(dr)
    if dc != 0:
        dc = dc // abs(dc)
    # Direction from 5 to 2s: (dr, dc)

    # Rotate 90 degrees counterclockwise: (dr, dc) -> (dc, -dr)
    new_dr, new_dc = dc, -dr

    # Place new 2s in the rotated direction from 5
    for i in range(1, count + 1):
        nr, nc = five_r + new_dr * i, five_c + new_dc * i
        if 0 <= nr < rows and 0 <= nc < cols:
            out[nr][nc] = 2

    # Original 2s become 3
    for r, c in twos:
        out[r][c] = 3

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
