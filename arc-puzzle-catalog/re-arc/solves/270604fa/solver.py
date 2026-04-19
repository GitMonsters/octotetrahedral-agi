import copy
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg_color = color_counts.most_common(1)[0][0]

    wall_color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color:
                wall_color = grid[r][c]
                break
        if wall_color is not None:
            break

    wall_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == wall_color]
    min_r = min(r for r, c in wall_cells)
    max_r = max(r for r, c in wall_cells)
    min_c = min(c for r, c in wall_cells)
    max_c = max(c for r, c in wall_cells)

    def find_gap(positions):
        return [(r, c) for r, c in positions if grid[r][c] == bg_color]

    sides = {
        'top':    [(min_r, c) for c in range(min_c, max_c + 1)],
        'bottom': [(max_r, c) for c in range(min_c, max_c + 1)],
        'left':   [(r, min_c) for r in range(min_r, max_r + 1)],
        'right':  [(r, max_c) for r in range(min_r, max_r + 1)],
    }

    gap_side = None
    gap_cells = []
    for side, positions in sides.items():
        g = find_gap(positions)
        if g:
            gap_side = side
            gap_cells = g
            break

    for r in range(min_r + 1, max_r):
        for c in range(min_c + 1, max_c):
            if grid[r][c] == bg_color:
                result[r][c] = 7

    for r, c in gap_cells:
        result[r][c] = 7

    if gap_side == 'top':
        gap_left = min(c for _, c in gap_cells)
        gap_right = max(c for _, c in gap_cells)
        for r in range(min_r - 1, -1, -1):
            for c in range(gap_left, gap_right + 1):
                result[r][c] = 7
        r, c = min_r - 1, gap_left - 1
        while r >= 0 and c >= 0:
            result[r][c] = 7
            r -= 1; c -= 1
        r, c = min_r - 1, gap_right + 1
        while r >= 0 and c < cols:
            result[r][c] = 7
            r -= 1; c += 1

    elif gap_side == 'bottom':
        gap_left = min(c for _, c in gap_cells)
        gap_right = max(c for _, c in gap_cells)
        for r in range(max_r + 1, rows):
            for c in range(gap_left, gap_right + 1):
                result[r][c] = 7
        r, c = max_r + 1, gap_left - 1
        while r < rows and c >= 0:
            result[r][c] = 7
            r += 1; c -= 1
        r, c = max_r + 1, gap_right + 1
        while r < rows and c < cols:
            result[r][c] = 7
            r += 1; c += 1

    elif gap_side == 'left':
        gap_top = min(r for r, _ in gap_cells)
        gap_bottom = max(r for r, _ in gap_cells)
        for c in range(min_c - 1, -1, -1):
            for r in range(gap_top, gap_bottom + 1):
                result[r][c] = 7
        r, c = gap_top - 1, min_c - 1
        while r >= 0 and c >= 0:
            result[r][c] = 7
            r -= 1; c -= 1
        r, c = gap_bottom + 1, min_c - 1
        while r < rows and c >= 0:
            result[r][c] = 7
            r += 1; c -= 1

    elif gap_side == 'right':
        gap_top = min(r for r, _ in gap_cells)
        gap_bottom = max(r for r, _ in gap_cells)
        for c in range(max_c + 1, cols):
            for r in range(gap_top, gap_bottom + 1):
                result[r][c] = 7
        r, c = gap_top - 1, max_c + 1
        while r >= 0 and c < cols:
            result[r][c] = 7
            r -= 1; c += 1
        r, c = gap_bottom + 1, max_c + 1
        while r < rows and c < cols:
            result[r][c] = 7
            r += 1; c += 1

    return result
