from collections import Counter


def _majority_color(grid):
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _bounding_box(grid, background):
    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != background
    ]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _border_color(grid, top, bottom, left, right):
    border = []
    for c in range(left, right + 1):
        border.append(grid[top][c])
        border.append(grid[bottom][c])
    for r in range(top + 1, bottom):
        border.append(grid[r][left])
        border.append(grid[r][right])
    return Counter(border).most_common(1)[0][0]


def _paint_color(grid, top, bottom, left, right, background, border):
    interior = [
        grid[r][c]
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
        if grid[r][c] not in (background, border)
    ]
    return Counter(interior).most_common(1)[0][0] if interior else border


def transform(grid):
    background = _majority_color(grid)
    top, bottom, left, right = _bounding_box(grid, background)
    border = _border_color(grid, top, bottom, left, right)
    color = _paint_color(grid, top, bottom, left, right, background, border)

    height = len(grid)
    width = len(grid[0])
    out = [row[:] for row in grid]

    row_sides = []
    if top == 0 and bottom != height - 1:
        row_sides.append((1, bottom))
    elif bottom == height - 1 and top != 0:
        row_sides.append((-1, top))
    else:
        row_sides.extend([(-1, top), (1, bottom)])

    col_sides = []
    if left == 0 and right != width - 1:
        col_sides.append((1, right))
    elif right == width - 1 and left != 0:
        col_sides.append((-1, left))
    else:
        col_sides.extend([(-1, left), (1, right)])

    for dr, r0 in row_sides:
        for dc, c0 in col_sides:
            r = r0 + dr
            c = c0 + dc
            while 0 <= r < height and 0 <= c < width:
                out[r][c] = color
                r += dr
                c += dc

    return out
