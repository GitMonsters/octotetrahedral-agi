"""Solver for ARC task 2dea5089

Rule: 4-fold symmetry reflection around a 2x2 center block.
- Find background color (most common)
- Find a 2x2 block of a non-background color (the center marker)
- Find scattered pixels of another non-background color (the pattern)
- Reflect all pattern pixels with 4-fold symmetry around the center of the 2x2 block
- Keep the 2x2 block unchanged
"""

from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    all_vals = [v for row in grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]

    # Find non-background pixels
    non_bg = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]

    if not non_bg:
        return [row[:] for row in grid]

    colors = set(non_bg.values())

    # Find which color forms a 2x2 block (the center marker)
    center_color = None
    center_r0, center_c0 = None, None
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] != bg and
                grid[r][c] == grid[r+1][c] == grid[r][c+1] == grid[r+1][c+1]):
                center_color = grid[r][c]
                center_r0, center_c0 = r, c
                break
        if center_color is not None:
            break

    # Pattern color is the other non-bg color
    pattern_color = None
    for color in colors:
        if color != center_color:
            pattern_color = color
            break

    if pattern_color is None:
        return [row[:] for row in grid]

    pattern_pixels = [(r, c) for (r, c), v in non_bg.items() if v == pattern_color]

    # Build output
    out = [[bg] * cols for _ in range(rows)]

    # Place the 2x2 center block
    out[center_r0][center_c0] = center_color
    out[center_r0][center_c0 + 1] = center_color
    out[center_r0 + 1][center_c0] = center_color
    out[center_r0 + 1][center_c0 + 1] = center_color

    # Reflect pattern pixels with 4-fold symmetry around center of 2x2 block
    for r, c in pattern_pixels:
        rr = 2 * center_r0 + 1 - r
        rc = 2 * center_c0 + 1 - c
        for nr, nc in [(r, c), (rr, c), (r, rc), (rr, rc)]:
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = pattern_color

    return out
