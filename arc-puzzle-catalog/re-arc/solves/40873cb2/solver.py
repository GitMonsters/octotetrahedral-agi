def transform(input_grid):
    """
    ARC Task 40873cb2

    Each adjacent pair of non-background cells spawns two 2x2 blocks of the
    partner's color, perpendicular to the pair axis at offset ±(2-3) cells.

    Direction rule:
    - Non-isolated pairs (component > 2 cells): blocks go INWARD toward the
      rest of the connected component.
    - Isolated pairs (component == 2 cells): direction depends on pair's row
      position relative to grid center:
        TOP half: V→LEFT, H→DOWN
        BOTTOM half: V→RIGHT, H→UP
    - Isolated single cells: one 2x2 block of own color placed diagonally
      (row away from center, col toward center).
    """
    H, W = len(input_grid), len(input_grid[0])
    grid = [row[:] for row in input_grid]
    center_row = (H - 1) / 2.0
    center_col = (W - 1) / 2.0

    # Find connected components of non-background cells
    visited = set()
    components = []
    comp_of = {}
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] == 3 or (r, c) in visited:
                continue
            comp = []
            queue = [(r, c)]
            visited.add((r, c))
            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc, input_grid[cr][cc]))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and input_grid[nr][nc] != 3:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            components.append(comp)
            for cr, cc, _ in comp:
                comp_of[(cr, cc)] = comp

    def place_block(r: int, c: int, color: int) -> None:
        for dr in range(2):
            for dc in range(2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    grid[nr][nc] = color

    # Process every adjacent pair of non-background cells
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] == 3:
                continue
            for orient, r2, c2 in [('H', r, c + 1), ('V', r + 1, c)]:
                if not (0 <= r2 < H and 0 <= c2 < W) or input_grid[r2][c2] == 3:
                    continue
                v1, v2 = input_grid[r][c], input_grid[r2][c2]
                comp = comp_of[(r, c)]
                comp_cells = set((cr, cc) for cr, cc, _ in comp)

                if len(comp) > 2:
                    rest = comp_cells - {(r, c), (r2, c2)}
                    if orient == 'V':
                        has_right = any(cc > c for _, cc in rest)
                        has_left = any(cc < c for _, cc in rest)
                        if has_right and not has_left:
                            direction = 'RIGHT'
                        elif has_left and not has_right:
                            direction = 'LEFT'
                        else:
                            direction = 'LEFT' if (r + r2) / 2 < center_row else 'RIGHT'
                    else:
                        has_below = any(rr > r for rr, _ in rest)
                        has_above = any(rr < r for rr, _ in rest)
                        if has_below and not has_above:
                            direction = 'DOWN'
                        elif has_above and not has_below:
                            direction = 'UP'
                        else:
                            direction = 'DOWN' if (r + r2) / 2 < center_row else 'UP'
                else:
                    pair_row = (r + r2) / 2.0
                    top_half = pair_row < center_row
                    if orient == 'V':
                        direction = 'LEFT' if top_half else 'RIGHT'
                    else:
                        direction = 'DOWN' if top_half else 'UP'

                if orient == 'V':
                    bc = c - 3 if direction == 'LEFT' else c + 2
                    place_block(r - 2, bc, v2)
                    place_block(r2 + 1, bc, v1)
                else:
                    br = r - 3 if direction == 'UP' else r + 2
                    place_block(br, c - 2, v2)
                    place_block(br, c2 + 1, v1)

    # Handle isolated single cells (component size 1)
    for comp in components:
        if len(comp) == 1:
            r, c, v = comp[0]
            br = r - 3 if r < center_row else r + 2
            bc = c - 3 if c > center_col else c + 2
            place_block(br, bc, v)

    return grid
