def transform(input_grid):
    from collections import deque

    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    bg = max(color_count, key=color_count.get)

    # Find connected components of non-background cells (8-connectivity)
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                comp = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc, grid[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                components.append(comp)

    # Classify: templates (size > 1) and standalone markers (size 1)
    templates = []
    standalone = []
    for comp in components:
        if len(comp) > 1:
            templates.append(comp)
        else:
            standalone.append(comp[0])

    # Extract template info: marker cell (color 0 or 8) and body offsets
    template_info = []
    for tmpl in templates:
        marker_cells = [(r, c, v) for r, c, v in tmpl if v in (0, 8)]
        if not marker_cells:
            continue
        mr, mc, mcolor = marker_cells[0]
        body = [(r - mr, c - mc, v) for r, c, v in tmpl if (r, c) != (mr, mc)]
        template_info.append({
            'marker_color': mcolor,
            'body_offsets': body,
        })

    # Stamp body at each standalone marker
    output = [row[:] for row in grid]
    for sr, sc, scolor in standalone:
        for tmpl in template_info:
            if scolor == tmpl['marker_color']:
                for dr, dc, bcolor in tmpl['body_offsets']:
                    if tmpl['marker_color'] == 8:
                        # Horizontally mirror (negate column offset)
                        nr, nc = sr + dr, sc - dc
                    else:
                        # Direct copy
                        nr, nc = sr + dr, sc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == bg:
                        output[nr][nc] = bcolor
                break

    return output
