"""
Solver for ARC task 52561d9e.

Pattern:
- There is a nested rectangular structure: optional border frame + optional padding + solid fill.
- The fill is the innermost solid rectangle of a non-background color.
- From each of the 4 corners of the fill rectangle, a diagonal ray extends outward
  (away from the rectangle center).
- The first 2 cells along each diagonal are skipped (they correspond to the
  border + padding layers, which may be explicit or virtual).
- After the 2-cell skip, the fill color is drawn on background cells until hitting
  the grid boundary.
"""

from collections import Counter


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Find background color (most frequent)
    color_counts: Counter = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg = color_counts.most_common(1)[0][0]

    # Find the fill: smallest solid rectangle among non-bg colors
    non_bg_colors = [c for c, _ in color_counts.most_common() if c != bg]

    fill_color = None
    fill_bbox = None
    min_area = float('inf')

    for color in non_bg_colors:
        min_r = min_c = float('inf')
        max_r = max_c = -1
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    count += 1
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        if area == count and area < min_area:
            min_area = area
            fill_color = color
            fill_bbox = (min_r, min_c, max_r, max_c)

    if fill_bbox is None:
        return out

    fr1, fc1, fr2, fc2 = fill_bbox

    # Diagonal directions from each corner (away from center)
    corners = [
        (fr1, fc1, -1, -1),  # top-left → up-left
        (fr1, fc2, -1, +1),  # top-right → up-right
        (fr2, fc1, +1, -1),  # bottom-left → down-left
        (fr2, fc2, +1, +1),  # bottom-right → down-right
    ]

    for cr, cc, dr, dc in corners:
        # Skip 2 cells from the fill corner
        r = cr + 3 * dr
        c = cc + 3 * dc
        # Draw fill color on background cells until grid boundary
        while 0 <= r < rows and 0 <= c < cols:
            if out[r][c] == bg:
                out[r][c] = fill_color
            r += dr
            c += dc

    return out
