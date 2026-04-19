import copy
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    counts = Counter()
    for r in range(rows):
        for c in range(cols):
            counts[grid[r][c]] += 1
    bg = counts.most_common(1)[0][0]

    result = copy.deepcopy(grid)

    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))

    for color, cells in color_cells.items():
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r][c] == color:
                    row_off = r - min_r
                    col_off = c - min_c
                    if row_off % 2 == 1 and col_off % 2 == 1:
                        result[r][c] = bg

    return result
