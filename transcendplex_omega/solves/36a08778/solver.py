def solve(grid):
    H = len(grid)
    W = len(grid[0])
    output = [row[:] for row in grid]

    # Find purple markers (color 6) and their columns
    purple_cols = set()
    start_row = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 6:
                purple_cols.add(c)
                start_row = max(start_row, r)

    if not purple_cols:
        return output

    # Find red bars (color 2) by row
    bars_by_row = {}
    for r in range(H):
        c = 0
        while c < W:
            if grid[r][c] == 2:
                start = c
                while c < W and grid[r][c] == 2:
                    c += 1
                bars_by_row.setdefault(r, []).append((start, c - 1))
            else:
                c += 1

    # Extending lines (set of columns currently going downward)
    extending = set(purple_cols)

    for r in range(start_row, H):
        # Mark extending columns as purple (skip red cells)
        for c in list(extending):
            if output[r][c] != 2:
                output[r][c] = 6

        # Check bars at next row
        if r + 1 < H and (r + 1) in bars_by_row:
            for s, e in bars_by_row[r + 1]:
                entering = {c for c in extending if s <= c <= e}
                if entering:
                    # Draw bracket top at this row
                    bl = max(0, s - 1)
                    br = min(W - 1, e + 1)
                    for c in range(bl, br + 1):
                        if output[r][c] != 2:
                            output[r][c] = 6
                    # Remove entering columns
                    extending -= entering
                    # Add bracket sides
                    if s - 1 >= 0:
                        extending.add(s - 1)
                    if e + 1 < W:
                        extending.add(e + 1)

    return output


if __name__ == '__main__':
    import json, sys
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train']):
        result = solve(ex['input'])
        print(f"Train {i}: {'PASS' if result == ex['output'] else 'FAIL'}")
        if result != ex['output']:
            diffs = [(r,c,result[r][c],ex['output'][r][c])
                     for r in range(len(ex['output'])) for c in range(len(ex['output'][0]))
                     if result[r][c] != ex['output'][r][c]]
            print(f"  {len(diffs)} diffs: {diffs[:10]}")
