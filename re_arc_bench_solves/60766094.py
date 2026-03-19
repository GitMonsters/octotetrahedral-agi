def transform(grid):
    rows, cols = len(grid), len(grid[0])
    from collections import Counter
    flat = [c for r in grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]

    visited = [[False]*cols for _ in range(rows)]
    objects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                stack = [(r, c)]
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr][cc] or grid[cr][cc] != color:
                        continue
                    visited[cr][cc] = True
                    cells.append((cr, cc))
                    stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                if cells:
                    objects.append((color, cells))

    corners = {}
    for color, cells in objects:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        top_count = sum(1 for r, c in cells if r == min_r)
        bot_count = sum(1 for r, c in cells if r == max_r)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        left_count = sum(1 for r, c in cells if c == min_c)
        right_count = sum(1 for r, c in cells if c == max_c)

        h_at_top = top_count >= bot_count
        h_arm = top_count if h_at_top else bot_count
        v_on_left = left_count >= right_count
        v_arm = left_count if v_on_left else right_count

        if h_at_top and v_on_left:
            corner = 'TL'
        elif h_at_top and not v_on_left:
            corner = 'TR'
        elif not h_at_top and v_on_left:
            corner = 'BL'
        else:
            corner = 'BR'
        corners[corner] = (color, h_arm, v_arm)

    tl_color, tl_h, tl_v = corners.get('TL', (bg, 0, 0))
    tr_color, tr_h, tr_v = corners.get('TR', (bg, 0, 0))
    bl_color, bl_h, bl_v = corners.get('BL', (bg, 0, 0))
    br_color, br_h, br_v = corners.get('BR', (bg, 0, 0))

    width = tl_h + tr_h
    height = tl_v + bl_v

    out = [[bg] * width for _ in range(height)]

    for c in range(tl_h):
        out[0][c] = tl_color
    for c in range(tl_h, width):
        out[0][c] = tr_color
    for c in range(bl_h):
        out[height - 1][c] = bl_color
    for c in range(bl_h, width):
        out[height - 1][c] = br_color
    for r in range(tl_v):
        out[r][0] = tl_color
    for r in range(tl_v, height):
        out[r][0] = bl_color
    for r in range(tr_v):
        out[r][width - 1] = tr_color
    for r in range(tr_v, height):
        out[r][width - 1] = br_color

    return out
