def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    ARC task 476d63ef:
    A single non-background pixel (dot) sits on an edge of the grid.
    From the dot's row downward to the grid bottom, draw alternating:
      - Full horizontal lines of the dot's color (at even offsets: 0, 2, 4, ...)
      - A single marker pixel (color 9) bouncing left/right (at odd offsets: 1, 3, 5, ...)
    The first marker appears on the opposite side from the dot, then alternates.
    """
    from collections import Counter

    H = len(grid)
    W = len(grid[0])

    # Detect background (most common) and dot (the single non-bg pixel)
    color_counts = Counter(v for row in grid for v in row)
    bg = color_counts.most_common(1)[0][0]

    dot_r = dot_c = dot_color = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                dot_r, dot_c, dot_color = r, c, grid[r][c]

    output = [row[:] for row in grid]

    marker_color = 9

    # Determine which side the dot is on (left=col 0, right=col W-1)
    if dot_c == W - 1:
        sides = [0, W - 1]  # first bounce goes to opposite (left), then back (right)
    else:
        sides = [W - 1, 0]  # dot on left edge: first bounce to right

    side_idx = 0
    for r in range(dot_r, H):
        offset = r - dot_r
        if offset % 2 == 0:
            # Horizontal line of dot_color
            for c in range(W):
                output[r][c] = dot_color
        else:
            # Marker on alternating side, rest stays bg
            output[r][sides[side_idx % 2]] = marker_color
            side_idx += 1

    return output
