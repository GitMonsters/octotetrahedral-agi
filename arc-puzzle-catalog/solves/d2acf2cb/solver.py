def solve(grid):
    """Find pairs of 4s in same row/column. Toggle cells between them: 0<->8, 6<->7.
    Column pairs take priority; remaining 4s form row pairs."""
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    toggle = {0: 8, 8: 0, 6: 7, 7: 6}

    from collections import defaultdict
    fours = set()
    row_fours = defaultdict(list)
    col_fours = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                fours.add((r, c))
                row_fours[r].append(c)
                col_fours[c].append(r)

    used = set()

    # Column pairs first (exactly 2 fours in column)
    for c, rs in col_fours.items():
        if len(rs) == 2:
            r1, r2 = sorted(rs)
            for r in range(r1 + 1, r2):
                if out[r][c] in toggle:
                    out[r][c] = toggle[out[r][c]]
            used.add((r1, c))
            used.add((r2, c))

    # Row pairs with remaining fours
    for r, cs in row_fours.items():
        remaining = sorted(c for c in cs if (r, c) not in used)
        if len(remaining) == 2:
            c1, c2 = remaining
            for c in range(c1 + 1, c2):
                if out[r][c] in toggle:
                    out[r][c] = toggle[out[r][c]]

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        status = "PASS" if result == ex['output'] else "FAIL"
        print(f"{'Train' if i < len(task['train']) else 'Test'} {i if i < len(task['train']) else i - len(task['train'])}: {status}")
