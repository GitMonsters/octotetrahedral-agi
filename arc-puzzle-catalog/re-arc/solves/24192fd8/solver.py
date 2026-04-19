def transform(input_grid):
    """
    Each input contains L-shaped pieces on a background. Each L is a corner
    fragment of a rectangle (top-left, top-right, bottom-left, bottom-right).
    The output assembles them into a single rectangle outline, placing each L
    at the appropriate corner with interior filled by background color.
    """
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Background = most common color
    bg = Counter(v for row in input_grid for v in row).most_common(1)[0][0]

    # Find connected components via same-color BFS
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in visited:
                color = input_grid[r][c]
                comp = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and (nr, nc) not in visited
                                and input_grid[nr][nc] == color):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append((color, comp))

    # Classify each L-shape by its corner type
    corners = {}  # 'TL'|'TR'|'BL'|'BR' -> (color, h_len, v_len)
    for color, comp in components:
        row_counts = Counter(r for r, c in comp)
        col_counts = Counter(c for r, c in comp)

        h_row = max(row_counts, key=row_counts.get)  # horizontal arm row
        v_col = max(col_counts, key=col_counts.get)   # vertical arm column

        h_cells = sorted(c for r, c in comp if r == h_row)
        v_cells = sorted(r for r, c in comp if c == v_col)
        h_len = len(h_cells)
        v_len = len(v_cells)

        h_dir = 'right' if v_col == min(h_cells) else 'left'
        v_dir = 'down' if h_row == min(v_cells) else 'up'

        if h_dir == 'right' and v_dir == 'down':
            ct = 'TL'
        elif h_dir == 'left' and v_dir == 'down':
            ct = 'TR'
        elif h_dir == 'right' and v_dir == 'up':
            ct = 'BL'
        else:
            ct = 'BR'

        corners[ct] = (color, h_len, v_len)

    # Compute output dimensions
    tl_h, tl_v = corners.get('TL', (0, 0, 0))[1], corners.get('TL', (0, 0, 0))[2]
    tr_h, tr_v = corners.get('TR', (0, 0, 0))[1], corners.get('TR', (0, 0, 0))[2]
    bl_h, bl_v = corners.get('BL', (0, 0, 0))[1], corners.get('BL', (0, 0, 0))[2]
    br_h, br_v = corners.get('BR', (0, 0, 0))[1], corners.get('BR', (0, 0, 0))[2]

    width = max(tl_h + tr_h, bl_h + br_h)
    height = max(tl_v + bl_v, tr_v + br_v)

    out = [[bg] * width for _ in range(height)]

    # Place each L at its corner
    if 'TL' in corners:
        c, h, v = corners['TL']
        for col in range(h):
            out[0][col] = c
        for row in range(v):
            out[row][0] = c

    if 'TR' in corners:
        c, h, v = corners['TR']
        for col in range(width - h, width):
            out[0][col] = c
        for row in range(v):
            out[row][width - 1] = c

    if 'BL' in corners:
        c, h, v = corners['BL']
        for col in range(h):
            out[height - 1][col] = c
        for row in range(height - v, height):
            out[row][0] = c

    if 'BR' in corners:
        c, h, v = corners['BR']
        for col in range(width - h, width):
            out[height - 1][col] = c
        for row in range(height - v, height):
            out[row][width - 1] = c

    return out
