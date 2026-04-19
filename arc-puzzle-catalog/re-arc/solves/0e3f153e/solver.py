import copy
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find all non-background cells
    non_bg = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg.append((r, c, grid[r][c]))

    non_bg_set = set((r, c) for r, c, _ in non_bg)

    # Find center: non-bg cell with all 4 diagonal neighbors non-bg
    # AND all 4 orthogonal neighbors are background (the X/diamond pattern)
    center = None
    for r, c, val in non_bg:
        if r < 1 or r >= rows - 1 or c < 1 or c >= cols - 1:
            continue
        diags_ok = all(
            (r + dr, c + dc) in non_bg_set
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        )
        orthos_ok = all(
            (r + dr, c + dc) not in non_bg_set
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
        if diags_ok and orthos_ok:
            center = (r, c)
            break

    out = copy.deepcopy(grid)

    # Apply 4-fold symmetry around the center
    cr, cc = center
    for r, c, val in non_bg:
        dr = r - cr
        dc = c - cc
        for sr, sc in [(dr, dc), (-dr, dc), (dr, -dc), (-dr, -dc)]:
            nr, nc = cr + sr, cc + sc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = val

    return out
