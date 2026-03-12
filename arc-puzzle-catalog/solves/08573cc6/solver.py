def solve(grid: list[list[int]]) -> list[list[int]]:
    """Draw a rectangular spiral from the 1-pixel center using two colors from top-left.
    
    Horizontal segments (left/right) use color1, vertical segments (down/up) use color2.
    Spiral starts going left with length 2, each subsequent segment increases by 1.
    Stops when a segment would extend beyond the grid boundary.
    """
    rows, cols = len(grid), len(grid[0])
    c1, c2 = grid[0][0], grid[0][1]

    # Find center (the 1-pixel)
    cr = cc = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                cr, cc = r, c

    out = [[0] * cols for _ in range(rows)]
    out[cr][cc] = 1

    # Directions: left, down, right, up
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    colors = [c1, c2, c1, c2]

    r, c = cr, cc
    seg_len = 2
    dir_idx = 0

    while True:
        dr, dc = dirs[dir_idx % 4]
        color = colors[dir_idx % 4]

        for _ in range(seg_len):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = color
                r, c = nr, nc
            else:
                return out  # Hit boundary → stop

        seg_len += 1
        dir_idx += 1

    return out


if __name__ == "__main__":
    import sys, json
    with open(sys.argv[1]) as f:
        task = json.load(f)
    for i, ex in enumerate(task['train'] + task['test']):
        result = solve(ex['input'])
        expected = ex.get('output')
        if expected:
            status = "PASS" if result == expected else "FAIL"
            print(f"Example {i}: {status}")
            if status == "FAIL":
                for r, (got, exp) in enumerate(zip(result, expected)):
                    if got != exp:
                        print(f"  Row {r}: got {got}")
                        print(f"       exp {exp}")
        else:
            print(f"Example {i}: no expected output")
