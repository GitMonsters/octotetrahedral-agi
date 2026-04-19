def transform(grid):
    rows, cols = len(grid), len(grid[0])

    # Find background color (most common)
    cc = {}
    for r in range(rows):
        for c in range(cols):
            cc[grid[r][c]] = cc.get(grid[r][c], 0) + 1
    bg = max(cc, key=cc.get)

    # Find connected components
    visited = [[False]*cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                queue = [(r, c)]
                visited[r][c] = True
                pixels = [(r, c)]
                while queue:
                    cr, cc2 = queue.pop(0)
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = cr+dr, cc2+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            pixels.append((nr, nc))
                            queue.append((nr, nc))
                min_r = min(r for r, c in pixels)
                max_r = max(r for r, c in pixels)
                min_c = min(c for r, c in pixels)
                max_c = max(c for r, c in pixels)
                components.append({
                    'color': color, 'pixels': pixels,
                    'min_r': min_r, 'max_r': max_r,
                    'min_c': min_c, 'max_c': max_c,
                    'h': max_r - min_r + 1, 'w': max_c - min_c + 1,
                    'area': len(pixels)
                })

    # Find corner markers: 4 single-pixel same-color components forming a rectangle
    markers = None
    marker_color = None
    singles_by_color = {}
    for comp in components:
        if comp['area'] == 1:
            singles_by_color.setdefault(comp['color'], []).append(comp)

    for color, singles in singles_by_color.items():
        if len(singles) >= 4:
            positions = [(s['min_r'], s['min_c']) for s in singles]
            unique_rows = sorted(set(r for r, c in positions))
            unique_cols = sorted(set(c for r, c in positions))
            if len(unique_rows) == 2 and len(unique_cols) == 2:
                corners = {(unique_rows[i], unique_cols[j]) for i in range(2) for j in range(2)}
                if corners.issubset(set(positions)):
                    markers = (unique_rows[0], unique_cols[0], unique_rows[1], unique_cols[1])
                    marker_color = color
                    break

    # Main blocks: large rectangular components, not marker color
    main_blocks = sorted(
        [c for c in components if c['area'] > 4 and c['color'] != marker_color],
        key=lambda b: (b['min_r'], b['min_c'])
    )
    block_color = main_blocks[0]['color'] if main_blocks else None

    # Template pixels: small non-marker components outside the marker rectangle
    template_pixels = []
    for comp in components:
        if comp['area'] <= 4 and comp['color'] != marker_color:
            for r, c in comp['pixels']:
                if markers:
                    mr1, mc1, mr2, mc2 = markers
                    if mr1 <= r <= mr2 and mc1 <= c <= mc2:
                        continue
                template_pixels.append((r, c, comp['color']))

    if not template_pixels:
        if markers:
            return [list(row[markers[1]:markers[3]+1]) for row in grid[markers[0]:markers[2]+1]]
        return [list(row) for row in grid]

    # Build template grid from the small pattern
    t_min_r = min(r for r, c, v in template_pixels)
    t_min_c = min(c for r, c, v in template_pixels)
    t_max_r = max(r for r, c, v in template_pixels)
    t_max_c = max(c for r, c, v in template_pixels)
    t_rows = t_max_r - t_min_r + 1
    t_cols = t_max_c - t_min_c + 1

    template = [[bg]*t_cols for _ in range(t_rows)]
    for r, c, v in template_pixels:
        template[r - t_min_r][c - t_min_c] = v

    # Classify template cells
    block_cells = sorted([(r, c) for r in range(t_rows) for c in range(t_cols)
                          if template[r][c] == block_color])
    new_cells = [(r, c, template[r][c]) for r in range(t_rows) for c in range(t_cols)
                 if template[r][c] != bg and template[r][c] != block_color]

    # Sub-block size
    if not main_blocks:
        return [row[:] for row in grid]
    if len(main_blocks) >= len(block_cells):
        sub_h = main_blocks[0]['h']
        sub_w = main_blocks[0]['w']
    else:
        block = main_blocks[0]
        bc_rows = [r for r, c in block_cells]
        bc_cols = [c for r, c in block_cells]
        bc_row_span = max(bc_rows) - min(bc_rows) + 1
        bc_col_span = max(bc_cols) - min(bc_cols) + 1
        sub_h = block['h'] // bc_row_span
        sub_w = block['w'] // bc_col_span

    # Arrangement start in input coordinates
    if not block_cells:
        return [row[:] for row in grid]
    if len(main_blocks) >= len(block_cells):
        bp = (main_blocks[0]['min_r'], main_blocks[0]['min_c'])
        tc = block_cells[0]
        arr_start_r = bp[0] - tc[0] * sub_h
        arr_start_c = bp[1] - tc[1] * sub_w
    else:
        block = main_blocks[0]
        bc_min_r = min(r for r, c in block_cells)
        bc_min_c = min(c for r, c in block_cells)
        tc = block_cells[0]
        sub_pos_r = block['min_r'] + (tc[0] - bc_min_r) * sub_h
        sub_pos_c = block['min_c'] + (tc[1] - bc_min_c) * sub_w
        arr_start_r = sub_pos_r - tc[0] * sub_h
        arr_start_c = sub_pos_c - tc[1] * sub_w

    # Build output
    if markers:
        mr1, mc1, mr2, mc2 = markers
        out_rows = mr2 - mr1 + 1
        out_cols = mc2 - mc1 + 1
        output = [list(row[mc1:mc2+1]) for row in grid[mr1:mr2+1]]

        arr_r = arr_start_r - mr1
        arr_c = arr_start_c - mc1

        for tr, tc_col, new_color in new_cells:
            nr = arr_r + tr * sub_h
            nc = arr_c + tc_col * sub_w
            for dr in range(sub_h):
                for dc in range(sub_w):
                    r, c = nr + dr, nc + dc
                    if 0 <= r < out_rows and 0 <= c < out_cols:
                        output[r][c] = new_color
        return output
    else:
        # No markers: construct output grid with padding from template gap
        block = main_blocks[0]
        row_gap = t_min_r - block['max_r'] - 1
        col_gap = t_min_c - block['max_c'] - 1

        total_padding = col_gap
        top_padding = row_gap
        bottom_padding = total_padding - top_padding
        left_padding = max(1, col_gap // sub_w)
        right_padding = total_padding - left_padding

        arr_h = t_rows * sub_h
        arr_w = t_cols * sub_w

        out_rows = top_padding + arr_h + bottom_padding
        out_cols = left_padding + arr_w + right_padding

        output = [[bg]*out_cols for _ in range(out_rows)]

        for tr in range(t_rows):
            for tc_col in range(t_cols):
                color = template[tr][tc_col]
                if color == bg:
                    continue
                nr = top_padding + tr * sub_h
                nc = left_padding + tc_col * sub_w
                for dr in range(sub_h):
                    for dc in range(sub_w):
                        r, c = nr + dr, nc + dc
                        if 0 <= r < out_rows and 0 <= c < out_cols:
                            output[r][c] = color
        return output
