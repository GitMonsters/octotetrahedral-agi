def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    flat = [input_grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find bounding box of foreground cells
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    height = max_r - min_r + 1
    width = max_c - min_c + 1
    unit_h = height // 3
    unit_w = width // 3

    # Reduce to 3x3 binary grid by checking top-left pixel of each block
    reduced = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = min_r + i * unit_h
            c = min_c + j * unit_w
            if input_grid[r][c] != bg:
                reduced[i][j] = 1

    # Check for hole: center empty with all 4 orthogonal neighbors filled
    has_hole = (
        reduced[1][1] == 0
        and reduced[0][1] == 1
        and reduced[1][0] == 1
        and reduced[1][2] == 1
        and reduced[2][1] == 1
    )

    return [[8]] if has_hole else [[2]]
