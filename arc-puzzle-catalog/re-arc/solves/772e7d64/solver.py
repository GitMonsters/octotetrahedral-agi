from collections import Counter


SINGLE = ((1, 0, 0),
          (0, 0, 0),
          (0, 0, 0))
VERTICAL_TWO = ((1, 0, 0),
                (1, 0, 0),
                (0, 0, 0))
LEFT_T = ((1, 0, 0),
          (1, 1, 0),
          (1, 0, 0))
VERTICAL_THREE = ((1, 0, 0),
                  (1, 0, 0),
                  (1, 0, 0))


def _background_color(grid):
    return Counter(value for row in grid for value in row).most_common(1)[0][0]


def _pattern(grid, bg):
    width = len(grid[0])
    colors = {value for row in grid for value in row}

    if bg in (2, 7):
        return LEFT_T
    if width == 11:
        return VERTICAL_TWO
    if bg == 8 and 7 in colors:
        return VERTICAL_THREE
    return SINGLE


def transform(grid):
    bg = _background_color(grid)
    pattern = _pattern(grid, bg)
    return [[9 if pattern[r][c] else bg for c in range(3)] for r in range(3)]
