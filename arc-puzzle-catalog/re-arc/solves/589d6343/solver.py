from collections import Counter


def _find_dividers(grid):
    row_div = None
    div_color = None
    for r, row in enumerate(grid):
        if len(set(row)) == 1:
            row_div = r
            div_color = row[0]
            break

    col_div = None
    for c in range(len(grid[0])):
        if all(grid[r][c] == div_color for r in range(len(grid))):
            col_div = c
            break

    return row_div, col_div, div_color


def _background_color(grid, row_div, col_div):
    cells = [
        grid[r][c]
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if r != row_div and c != col_div
    ]
    return Counter(cells).most_common(1)[0][0]


def transform(grid):
    row_div, col_div, div_color = _find_dividers(grid)
    bg = _background_color(grid, row_div, col_div)

    quadrants = [
        (0, row_div, 0, col_div),
        (0, row_div, col_div + 1, len(grid[0])),
        (row_div + 1, len(grid), 0, col_div),
        (row_div + 1, len(grid), col_div + 1, len(grid[0])),
    ]

    def score(bounds):
        r0, r1, c0, c1 = bounds
        return sum(grid[r][c] != bg for r in range(r0, r1) for c in range(c0, c1))

    source = max(quadrants, key=score)
    sr0, sr1, sc0, sc1 = source
    pattern = [row[sc0:sc1] for row in grid[sr0:sr1]]

    out = [row[:] for row in grid]
    for r0, r1, c0, c1 in quadrants:
        for rr, r in enumerate(range(r0, r1)):
            for cc, c in enumerate(range(c0, c1)):
                out[r][c] = pattern[rr][cc]

    return out
