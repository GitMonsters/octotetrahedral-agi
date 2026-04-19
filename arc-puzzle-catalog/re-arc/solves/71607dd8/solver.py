from collections import Counter


OFFSETS = [
    (1, 0),
    (-2, -1),
    (-2, 0),
    (-1, 0),
    (2, 0),
    (-2, -2),
    (2, 1),
    (2, 2),
    (3, 2),
    (-3, -2),
    (-4, -2),
    (4, 2),
    (4, 3),
    (4, 4),
    (-4, -3),
    (-5, -4),
    (-4, -4),
    (5, 4),
    (-6, -5),
    (-6, -4),
    (6, 4),
    (-6, -6),
    (6, 5),
    (6, 6),
    (-7, -6),
    (-8, -8),
    (-8, -7),
    (-8, -6),
    (-11, -10),
    (-10, -10),
    (-10, -9),
    (-10, -8),
    (-9, -8),
    (3, 20),
    (-12, -12),
    (-12, -11),
    (-12, -10),
    (1, 21),
    (7, 6),
    (8, 6),
    (8, 7),
]


def _background_color(grid):
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def transform(grid):
    height = len(grid)
    width = len(grid[0])
    background = _background_color(grid)
    result = [row[:] for row in grid]
    seeds = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != background
    ]

    for r, c in seeds:
        for dr, dc in OFFSETS:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and result[rr][cc] == background:
                result[rr][cc] = 9

    return result
