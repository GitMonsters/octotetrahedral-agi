def solve(grid: list[list[int]]) -> list[list[int]]:
    """Find rectangles, match scattered colors by count, fill interiors with checkerboard.
    
    Pattern: Each rectangle border is drawn with one color. Scattered dots of other colors
    appear outside. The count of scattered dots of color C equals the number of checkerboard
    cells in the rectangle that C should fill. Fill each rectangle's interior with a
    checkerboard of its matched color, then clear everything else to 0.
    """
    from collections import defaultdict

    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Group non-zero cells by color
    color_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].add((r, c))

    # Identify colors that form rectangle borders
    rects: list[tuple[int, int, int, int, int]] = []  # (color, r1, r2, c1, c2)
    rect_colors: set[int] = set()
    rect_cell_set: set[tuple[int, int]] = set()

    for color, cells in color_cells.items():
        if len(cells) < 4:
            continue
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Build expected border cells
        expected: set[tuple[int, int]] = set()
        for cc in range(min_c, max_c + 1):
            expected.add((min_r, cc))
            expected.add((max_r, cc))
        for rr in range(min_r, max_r + 1):
            expected.add((rr, min_c))
            expected.add((rr, max_c))

        if cells == expected:
            rects.append((color, min_r, max_r, min_c, max_c))
            rect_colors.add(color)
            rect_cell_set.update(cells)

    # Count scattered (non-rectangle) colored cells
    scatter_counts: dict[int, int] = defaultdict(int)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in rect_cell_set:
                scatter_counts[grid[r][c]] += 1

    # For each rectangle, compute its checkerboard fill count
    rect_checker: dict[tuple[int, int, int, int, int], int] = {}
    for rect in rects:
        color, r1, r2, c1, c2 = rect
        int_h = r2 - r1 - 1
        int_w = c2 - c1 - 1
        if int_h <= 0 or int_w <= 0:
            continue
        count = sum(1 for dr in range(int_h) for dc in range(int_w) if (dr + dc) % 2 == 0)
        rect_checker[rect] = count

    # Match: scattered color count == rectangle checkerboard count
    used_colors: set[int] = set()
    rect_fill: dict[tuple[int, int, int, int, int], int] = {}
    for rect, checker_count in rect_checker.items():
        for sc_color, sc_count in scatter_counts.items():
            if sc_count == checker_count and sc_color not in used_colors:
                rect_fill[rect] = sc_color
                used_colors.add(sc_color)
                break

    # Build output: place rectangle borders
    for color, r1, r2, c1, c2 in rects:
        for cc in range(c1, c2 + 1):
            result[r1][cc] = color
            result[r2][cc] = color
        for rr in range(r1, r2 + 1):
            result[rr][c1] = color
            result[rr][c2] = color

    # Fill interiors with checkerboard
    for rect, fill_color in rect_fill.items():
        _, r1, r2, c1, c2 = rect
        for dr in range(r2 - r1 - 1):
            for dc in range(c2 - c1 - 1):
                if (dr + dc) % 2 == 0:
                    result[r1 + 1 + dr][c1 + 1 + dc] = fill_color

    return result
