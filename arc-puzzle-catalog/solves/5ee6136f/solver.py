def transform(grid):
    from collections import Counter
    rows = len(grid)
    cols = len(grid[0])
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    out = [row[:] for row in grid]

    # Each non-bg cell creates a vertical zigzag pattern
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                color = grid[r][c]
                for row in range(rows):
                    d = abs(row - r)
                    if d % 2 == 0:
                        out[row][c] = color
                    else:
                        if c - 1 >= 0:
                            out[row][c - 1] = color
                        if c + 1 < cols:
                            out[row][c + 1] = color

    return out
