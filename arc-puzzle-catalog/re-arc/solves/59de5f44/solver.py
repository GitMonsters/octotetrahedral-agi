from collections import Counter
from typing import List, Tuple

Grid = List[List[int]]


def transform(grid: Grid) -> Grid:
    counts = Counter(value for row in grid for value in row)
    background = counts.most_common(1)[0][0]

    points: List[Tuple[int, int, int]] = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != background:
                points.append((c, r, value))

    points.sort()
    order = [
        (0, 0), (0, 1), (0, 2),
        (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2),
    ]

    out = [[background] * 3 for _ in range(3)]
    start = 9 - len(points)
    for (_, _, value), (orow, ocol) in zip(points, order[start:]):
        out[orow][ocol] = value
    return out
