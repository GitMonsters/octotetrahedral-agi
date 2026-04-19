def fill_row(grid, row, value):
    for col in range(len(grid[0])):
        grid[row][col] = value


def fill_col(grid, col, value):
    for row in range(len(grid)):
        grid[row][col] = value


def transform(grid):
    out = [row[:] for row in grid]
    rows, cols = len(out), len(out[0])
    background = out[0][0]
    markers = [(r, c, out[r][c]) for r in range(rows) for c in range(cols) if out[r][c] != background]

    if not markers:
        return out

    marker_rows = {r for r, _, _ in markers}

    if len(marker_rows) == 1 and len(markers) >= 2:
        ordered = sorted(markers, key=lambda item: item[1])
        anchor_col = ordered[0][1]
        for _, col, value in ordered:
            fill_col(out, col, value)
        for _, col, value in ordered[1:]:
            reflected_col = 2 * anchor_col - col
            if 0 <= reflected_col < cols:
                fill_col(out, reflected_col, value)
        return out

    for row, _, value in markers:
        fill_row(out, row, value)
    return out
