from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Background = most frequent color
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find horizontal line: a row entirely one non-bg color
    h_row = None
    line_color = None
    for r in range(rows):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != bg:
            h_row = r
            line_color = grid[r][0]
            break

    # Find vertical line: a column entirely the line color
    v_col = None
    for c in range(cols):
        if all(grid[r][c] == line_color for r in range(rows)):
            v_col = c
            break

    # Find noise pixels: not background and not on the cross
    noise = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not (r == h_row or c == v_col):
                noise.append((r, c))

    N = len(noise)
    if N == 0:
        return [row[:] for row in grid]

    # Quadrant of noise determines opposite shift direction
    nr, nc = noise[0]
    dr = -1 if nr > h_row else 1
    dc = -1 if nc > v_col else 1

    new_h = h_row + N * dr
    new_v = v_col + N * dc

    # Build clean output with shifted cross
    out = [[bg] * cols for _ in range(rows)]
    for c in range(cols):
        out[new_h][c] = line_color
    for r in range(rows):
        out[r][new_v] = line_color

    return out
