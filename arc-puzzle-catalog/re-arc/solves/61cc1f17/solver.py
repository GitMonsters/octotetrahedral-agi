from collections import Counter
from typing import List


def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    signal_cells = [
        (r, c, grid[r][c])
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] != bg
    ]
    if len(signal_cells) != 1:
        return [row[:] for row in grid]

    sr, sc, sig_color = signal_cells[0]

    if sc == 0:
        signal_edge_col = 0
        far_edge_col = cols - 1
    else:
        signal_edge_col = cols - 1
        far_edge_col = 0

    result = [row[:] for row in grid]

    for r in range(sr + 1):
        dist = sr - r
        if dist % 2 == 0:
            result[r] = [sig_color] * cols
        else:
            result[r] = [bg] * cols
            if dist % 4 == 1:
                result[r][far_edge_col] = 8
            else:
                result[r][signal_edge_col] = 8

    return result
