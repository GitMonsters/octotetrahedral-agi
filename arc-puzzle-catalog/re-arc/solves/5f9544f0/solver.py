def transform(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: Find the large rectangle filled mostly with background color.
    Inside it, sparse "noise" pixels of a single color mark row/column crosses.
    Output = rectangle dimensions, with full rows and columns drawn through
    each noise pixel position using the noise color. Everything else is background.
    """
    from collections import Counter

    grid = input_grid
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    color_counts: Counter[int] = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    bg_color = color_counts.most_common(1)[0][0]

    # Prefix sum of background cells for O(1) rectangle queries
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            prefix[r + 1][c + 1] = (
                prefix[r][c + 1] + prefix[r + 1][c]
                - prefix[r][c]
                + (1 if grid[r][c] == bg_color else 0)
            )

    def rect_bg(r1: int, c1: int, r2: int, c2: int) -> int:
        return (
            prefix[r2 + 1][c2 + 1]
            - prefix[r1][c2 + 1]
            - prefix[r2 + 1][c1]
            + prefix[r1][c1]
        )

    # Find largest rectangle whose 4 boundary edges are entirely background
    # and whose interior is ≥85% background (noise pixels are sparse)
    best = None
    best_area = 0
    min_dim = 5

    for r1 in range(rows):
        for r2 in range(r1 + min_dim - 1, rows):
            h = r2 - r1 + 1
            for c1 in range(cols):
                for c2 in range(c1 + min_dim - 1, cols):
                    w = c2 - c1 + 1
                    area = h * w
                    if area <= best_area:
                        continue
                    # All 4 boundary edges must be pure background
                    if rect_bg(r1, c1, r1, c2) != w:
                        continue
                    if rect_bg(r2, c1, r2, c2) != w:
                        continue
                    if rect_bg(r1, c1, r2, c1) != h:
                        continue
                    if rect_bg(r1, c2, r2, c2) != h:
                        continue
                    if rect_bg(r1, c1, r2, c2) / area < 0.85:
                        continue
                    best_area = area
                    best = (r1, c1, r2, c2)

    r1, c1, r2, c2 = best  # type: ignore[misc]
    rect_h = r2 - r1 + 1
    rect_w = c2 - c1 + 1

    # Collect noise pixel positions and color
    noise_rows: set[int] = set()
    noise_cols: set[int] = set()
    noise_color = None
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if grid[r][c] != bg_color:
                noise_rows.add(r - r1)
                noise_cols.add(c - c1)
                noise_color = grid[r][c]

    # Build output: background + full-row/full-column crosses at noise positions
    output = [[bg_color] * rect_w for _ in range(rect_h)]
    if noise_color is not None:
        for r in range(rect_h):
            for c in range(rect_w):
                if r in noise_rows or c in noise_cols:
                    output[r][c] = noise_color

    return output
