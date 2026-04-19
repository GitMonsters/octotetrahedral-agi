def transform(input_grid):
    """
    Rule: Nested rectangles emit diagonal rays from the outermost frame's corners.
    - Find background (most common color)
    - Find outermost bbox of all non-bg cells (the outer frame)
    - Peel inward to find the innermost filled region
    - If an inner region exists, draw diagonals from the 4 outermost bbox corners
      outward at 45°, using the innermost region's color
    - If no inner region (just a hollow frame), no change
    """
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter
    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] += 1
    bg = color_count.most_common(1)[0][0]

    non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not non_bg:
        return grid

    outer_r1 = min(r for r, c in non_bg)
    outer_r2 = max(r for r, c in non_bg)
    outer_c1 = min(c for r, c in non_bg)
    outer_c2 = max(c for r, c in non_bg)

    def find_innermost_color(r1: int, c1: int, r2: int, c2: int) -> int | None:
        inner = [(r, c) for r in range(r1 + 1, r2)
                         for c in range(c1 + 1, c2)
                         if grid[r][c] != bg]
        if not inner:
            return None

        ir1 = min(r for r, c in inner)
        ir2 = max(r for r, c in inner)
        ic1 = min(c for r, c in inner)
        ic2 = max(c for r, c in inner)

        is_fill = all(grid[r][c] != bg
                      for r in range(ir1, ir2 + 1)
                      for c in range(ic1, ic2 + 1))
        if is_fill:
            return grid[ir1][ic1]

        deeper = find_innermost_color(ir1, ic1, ir2, ic2)
        return deeper if deeper is not None else grid[ir1][ic1]

    innermost = find_innermost_color(outer_r1, outer_c1, outer_r2, outer_c2)
    if innermost is None:
        return grid

    corners = [(outer_r1, outer_c1), (outer_r1, outer_c2),
               (outer_r2, outer_c1), (outer_r2, outer_c2)]
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for (sr, sc), (dr, dc) in zip(corners, directions):
        r, c = sr + dr, sc + dc
        while 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = innermost
            r += dr
            c += dc

    return grid
