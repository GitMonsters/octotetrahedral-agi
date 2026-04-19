from collections import Counter


def dominant_color(grid):
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def trace_bouncing_diagonal(out, row, col, color):
    height = len(out)
    width = len(out[0])

    horizontal_major = width >= height

    if horizontal_major:
        dcol = 1 if col == 0 else -1 if col == width - 1 else 1
        drow = 1 if row == 0 else -1 if row == height - 1 else 1
        while 0 <= col < width:
            out[row][col] = color
            next_col = col + dcol
            if not 0 <= next_col < width:
                break
            next_row = row + drow
            if next_row < 0 or next_row >= height:
                drow = -drow
                next_row = row + drow
            row, col = next_row, next_col
        return

    drow = 1 if row == 0 else -1 if row == height - 1 else 1
    dcol = 1 if col == 0 else -1 if col == width - 1 else 1
    while 0 <= row < height:
        out[row][col] = color
        next_row = row + drow
        if not 0 <= next_row < height:
            break
        next_col = col + dcol
        if next_col < 0 or next_col >= width:
            dcol = -dcol
            next_col = col + dcol
        row, col = next_row, next_col


def transform(grid):
    background = dominant_color(grid)
    sources = [
        (row_index, col_index, value)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value != background
    ]
    if not sources:
        return [row[:] for row in grid]

    out = [[background for _ in row] for row in grid]
    for row, col, color in sources:
        trace_bouncing_diagonal(out, row, col, color)
    return out
