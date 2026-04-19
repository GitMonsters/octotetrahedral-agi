from collections import Counter
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])

    # Identify background (most common color)
    color_counts: Counter = Counter()
    for row in grid:
        color_counts.update(row)
    bg = color_counts.most_common(1)[0][0]

    # Group non-background pixels by color
    pixel_groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                pixel_groups.setdefault(v, []).append((r, c))

    # If no non-bg pixels, return as-is
    if not pixel_groups:
        return [row[:] for row in grid]

    # Find the X marker: 5 pixels forming center + 4 diagonal neighbors
    x_center = None
    x_color = None
    for color, pixels in pixel_groups.items():
        if len(pixels) == 5:
            ps = set(pixels)
            for r, c in pixels:
                diags = {(r - 1, c - 1), (r - 1, c + 1),
                         (r + 1, c - 1), (r + 1, c + 1)}
                if diags.issubset(ps):
                    x_center = (r, c)
                    x_color = color
                    break
        if x_center is not None:
            break

    # If no X marker found, return as-is
    if x_center is None:
        return [row[:] for row in grid]

    cr, cc = x_center

    # Collect all shape pixels (non-bg, non-X-color)
    shape_pixels: list[tuple[int, int, int]] = []
    for color, pixels in pixel_groups.items():
        if color == x_color:
            continue
        for r, c in pixels:
            shape_pixels.append((r, c, color))

    # If no shape, return as-is
    if not shape_pixels:
        return [row[:] for row in grid]

    # Build output: start with background, keep X
    out = [[bg] * cols for _ in range(rows)]
    for r, c in pixel_groups[x_color]:
        out[r][c] = x_color

    # Place 4-fold reflections of the shape around X center
    for r, c, color in shape_pixels:
        dr = r - cr
        dc = c - cc
        for sr, sc in [(dr, dc), (-dr, dc), (dr, -dc), (-dr, -dc)]:
            nr, nc = cr + sr, cc + sc
            if 0 <= nr < rows and 0 <= nc < cols:
                if out[nr][nc] == bg:
                    out[nr][nc] = color

    return out
