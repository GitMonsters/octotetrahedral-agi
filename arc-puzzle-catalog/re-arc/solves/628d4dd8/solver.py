from collections import Counter


def _majority_color(values):
    return Counter(values).most_common(1)[0][0]


def _horizontal_fill(grid):
    return [[_majority_color(row)] * len(row) for row in grid]


def _vertical_fill(grid):
    h = len(grid)
    w = len(grid[0])
    cols = [_majority_color(grid[i][j] for i in range(h)) for j in range(w)]
    return [cols[:] for _ in range(h)]


def _match_score(grid, candidate):
    return sum(
        1
        for i, row in enumerate(grid)
        for j, value in enumerate(row)
        if value == candidate[i][j]
    )


def transform(grid):
    horizontal = _horizontal_fill(grid)
    vertical = _vertical_fill(grid)
    if _match_score(grid, horizontal) >= _match_score(grid, vertical):
        return horizontal
    return vertical
