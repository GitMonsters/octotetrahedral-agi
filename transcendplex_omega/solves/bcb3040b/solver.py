def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find the two 2s
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    (r1, c1), (r2, c2) = twos

    # Walk from (r1,c1) to (r2,c2) in unit steps
    dr = (0 if r2 == r1 else (1 if r2 > r1 else -1))
    dc = (0 if c2 == c1 else (1 if c2 > c1 else -1))

    r, c = r1, c1
    while True:
        if out[r][c] == 0:
            out[r][c] = 2
        elif out[r][c] == 1:
            out[r][c] = 3
        if r == r2 and c == c2:
            break
        r += dr
        c += dc

    return out
