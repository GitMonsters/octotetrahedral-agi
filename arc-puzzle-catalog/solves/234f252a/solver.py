def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Each input contains 3-4 L-shaped pieces scattered on a background.
    Each L-shape has a corner orientation (TL/TR/BL/BR) determined by
    where its two perpendicular arms meet. The output assembles all
    L-shapes into a single rectangular frame, placing each at its
    matching corner.
    """
    from collections import Counter, deque

    H = len(grid)
    W = len(grid[0])

    # Find background color (most common)
    color_count: Counter[int] = Counter()
    for r in range(H):
        for c in range(W):
            color_count[grid[r][c]] += 1
    bg = color_count.most_common(1)[0][0]

    # Find connected components via 4-connected BFS
    visited = [[False] * W for _ in range(H)]
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                queue = deque([(r, c)])
                visited[r][c] = True
                cells: list[tuple[int, int]] = []
                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append((color, cells))

    # Analyze each L-shaped component
    pieces: dict[str, tuple[int, int, int]] = {}  # corner -> (color, h_arm_len, v_arm_len)

    for color, cells in components:
        rows_of = [r for r, c in cells]
        cols_of = [c for r, c in cells]
        min_r, max_r = min(rows_of), max(rows_of)
        min_c, max_c = min(cols_of), max(cols_of)

        row_counts: Counter[int] = Counter(rows_of)
        col_counts: Counter[int] = Counter(cols_of)

        h_arm_row = max(row_counts, key=lambda k: row_counts[k])
        h_arm_len = row_counts[h_arm_row]

        v_arm_col = max(col_counts, key=lambda k: col_counts[k])
        v_arm_len = col_counts[v_arm_col]

        # Determine corner type from where the two arms intersect
        if h_arm_row == min_r and v_arm_col == min_c:
            corner = 'TL'
        elif h_arm_row == min_r and v_arm_col == max_c:
            corner = 'TR'
        elif h_arm_row == max_r and v_arm_col == min_c:
            corner = 'BL'
        else:
            corner = 'BR'

        pieces[corner] = (color, h_arm_len, v_arm_len)

    # Determine output dimensions from matching edge pairs
    width = None
    height = None

    if 'TL' in pieces and 'TR' in pieces:
        width = pieces['TL'][1] + pieces['TR'][1]
    if 'BL' in pieces and 'BR' in pieces:
        w = pieces['BL'][1] + pieces['BR'][1]
        width = w if width is None else width
    if 'TL' in pieces and 'BL' in pieces:
        height = pieces['TL'][2] + pieces['BL'][2]
    if 'TR' in pieces and 'BR' in pieces:
        h = pieces['TR'][2] + pieces['BR'][2]
        height = h if height is None else height

    assert width is not None and height is not None, "Cannot determine output dimensions"

    # Build output grid
    out = [[bg] * width for _ in range(height)]

    for corner, (color, h, v) in pieces.items():
        if corner == 'TL':
            for c in range(h):
                out[0][c] = color
            for r in range(v):
                out[r][0] = color
        elif corner == 'TR':
            for c in range(width - h, width):
                out[0][c] = color
            for r in range(v):
                out[r][width - 1] = color
        elif corner == 'BL':
            for c in range(h):
                out[height - 1][c] = color
            for r in range(height - v, height):
                out[r][0] = color
        elif corner == 'BR':
            for c in range(width - h, width):
                out[height - 1][c] = color
            for r in range(height - v, height):
                out[r][width - 1] = color

    return out
