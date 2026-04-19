def transform(input_grid):
    import numpy as np
    from collections import defaultdict
    from itertools import combinations

    grid = np.array(input_grid)
    rows, cols = grid.shape
    bg = int(np.bincount(grid.flatten()).argmax())

    # Find connected components (4-connected)
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r, c] and grid[r, c] != bg:
                color = int(grid[r, c])
                stack = [(r, c)]
                comp = set()
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < rows and 0 <= cc < cols and not visited[cr, cc] and grid[cr, cc] == color:
                        visited[cr, cc] = True
                        comp.add((cr, cc))
                        stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])
                components.append((color, comp))

    # Identify corner markers: 4 size-1 components of same color forming rectangle
    singles_by_color = defaultdict(list)
    for color, comp in components:
        if len(comp) == 1:
            singles_by_color[color].append(list(comp)[0])

    corner_color = None
    corner_positions = None
    rect_r1 = rect_r2 = rect_c1 = rect_c2 = None
    best_area = -1

    for color, positions in singles_by_color.items():
        if len(positions) >= 4:
            for combo in combinations(positions, 4):
                rs = sorted(set(p[0] for p in combo))
                cs = sorted(set(p[1] for p in combo))
                if len(rs) == 2 and len(cs) == 2:
                    expected = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
                    if set(combo) == expected:
                        area = (rs[1] - rs[0]) * (cs[1] - cs[0])
                        if area > best_area:
                            best_area = area
                            corner_color = color
                            corner_positions = expected
                            rect_r1, rect_r2 = rs[0], rs[1]
                            rect_c1, rect_c2 = cs[0], cs[1]

    # Classify components as inside blocks or key pixels
    inside_blocks = []  # (color, rel_row, rel_col, height, width)
    key_pixels = []     # (color, abs_row, abs_col)

    for color, comp in components:
        if len(comp) == 1 and list(comp)[0] in corner_positions:
            continue

        min_r = min(p[0] for p in comp)
        max_r = max(p[0] for p in comp)
        min_c = min(p[1] for p in comp)
        max_c = max(p[1] for p in comp)

        if rect_r1 <= min_r and max_r <= rect_r2 and rect_c1 <= min_c and max_c <= rect_c2:
            inside_blocks.append((color, min_r - rect_r1, min_c - rect_c1,
                                  max_r - min_r + 1, max_c - min_c + 1))
        else:
            for pixel in comp:
                key_pixels.append((color, pixel[0], pixel[1]))

    block_h = inside_blocks[0][3]
    block_w = inside_blocks[0][4]

    # Build key grid (each pixel maps to one block-sized region)
    key_min_r = min(p[1] for p in key_pixels)
    key_min_c = min(p[2] for p in key_pixels)
    key_grid = {}
    for color, r, c in key_pixels:
        key_grid[(r - key_min_r, c - key_min_c)] = color

    # Find block grid origin by matching inside blocks to key positions
    candidate_origins = defaultdict(int)
    for bcolor, rel_r, rel_c, h, w in inside_blocks:
        for (kr, kc), kcolor in key_grid.items():
            if kcolor == bcolor:
                origin_r = rel_r - kr * block_h
                origin_c = rel_c - kc * block_w
                candidate_origins[(origin_r, origin_c)] += 1

    origin_r, origin_c = max(candidate_origins, key=candidate_origins.get)

    # Build output (rectangle size, filled with background)
    out_h = rect_r2 - rect_r1 + 1
    out_w = rect_c2 - rect_c1 + 1
    output = [[bg] * out_w for _ in range(out_h)]

    # Place blocks according to key layout
    for (kr, kc), kcolor in key_grid.items():
        br = origin_r + kr * block_h
        bc = origin_c + kc * block_w
        for dr in range(block_h):
            for dc in range(block_w):
                r, c = br + dr, bc + dc
                if 0 <= r < out_h and 0 <= c < out_w:
                    output[r][c] = kcolor

    # Place corner markers at output corners
    output[0][0] = corner_color
    output[0][out_w - 1] = corner_color
    output[out_h - 1][0] = corner_color
    output[out_h - 1][out_w - 1] = corner_color

    return output
