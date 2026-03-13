def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find the 2-bordered rectangle with a 2x2 quadrant color pattern inside,
    then scale that pattern to fill the entire interior."""
    rows, cols = len(grid), len(grid[0])

    # Find bounding box of the rectangle (border color = 2)
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)

    # Interior bounds
    int_top, int_bot = min_r + 1, max_r - 1
    int_left, int_right = min_c + 1, max_c - 1
    int_h = int_bot - int_top + 1
    int_w = int_right - int_left + 1

    # Find colored cells inside the rectangle
    colored = [(r, c, grid[r][c])
               for r in range(int_top, int_bot + 1)
               for c in range(int_left, int_right + 1)
               if grid[r][c] not in (0, 2)]

    # Bounding box of colored pattern
    cr_min = min(r for r, c, v in colored)
    cc_min = min(c for r, c, v in colored)
    cr_max = max(r for r, c, v in colored)
    cc_max = max(c for r, c, v in colored)
    pat_h = cr_max - cr_min + 1
    pat_w = cc_max - cc_min + 1

    # Extract 2x2 quadrant colors
    mid_r = cr_min + pat_h // 2
    mid_c = cc_min + pat_w // 2
    tl = grid[cr_min][cc_min]
    tr = grid[cr_min][mid_c]
    bl = grid[mid_r][cc_min]
    br = grid[mid_r][mid_c]

    # Build output: border + scaled interior
    out_h, out_w = int_h + 2, int_w + 2
    half_h, half_w = int_h // 2, int_w // 2
    result = [[2] * out_w for _ in range(out_h)]

    for r in range(int_h):
        for c in range(int_w):
            color = (tl if r < half_h else bl) if c < half_w else (tr if r < half_h else br)
            result[r + 1][c + 1] = color

    return result
