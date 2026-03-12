def solve(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = [[0] * cols for _ in range(rows)]

    # Extract color palette: non-zero, non-8 values left-to-right from any row
    palette = []
    for c in range(cols):
        val = grid[0][c]
        if val != 0 and val != 8:
            palette.append(val)
        elif val == 8:
            break

    # Greedy rectangle decomposition of 8-cells
    # Scan columns left-to-right, rows top-to-bottom
    processed = [[False] * cols for _ in range(rows)]
    rectangles = []

    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == 8 and not processed[r][c]:
                # Extend down
                r_end = r
                while r_end + 1 < rows and grid[r_end + 1][c] == 8 and not processed[r_end + 1][c]:
                    r_end += 1
                # Extend right
                c_end = c
                while c_end + 1 < cols:
                    ok = all(
                        grid[rr][c_end + 1] == 8 and not processed[rr][c_end + 1]
                        for rr in range(r, r_end + 1)
                    )
                    if ok:
                        c_end += 1
                    else:
                        break
                # Mark processed
                for rr in range(r, r_end + 1):
                    for cc in range(c, c_end + 1):
                        processed[rr][cc] = True
                rectangles.append((r, c, r_end, c_end))

    # Assign palette colors to rectangles in discovery order
    for i, (r1, c1, r2, c2) in enumerate(rectangles):
        color = palette[i] if i < len(palette) else 0
        for rr in range(r1, r2 + 1):
            for cc in range(c1, c2 + 1):
                output[rr][cc] = color

    return output
