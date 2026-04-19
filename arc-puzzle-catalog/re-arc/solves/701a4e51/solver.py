def transform(grid):
    import copy
    h = len(grid)
    w = len(grid[0])
    out = copy.deepcopy(grid)

    # Find background (most common value)
    from collections import Counter
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find all non-bg colors
    all_colors = set(flat) - {bg}

    # Find connected components (8-connected)
    visited = [[False]*w for _ in range(h)]
    groups = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                comp = [(r, c)]
                while queue:
                    cr, cc = queue.pop(0)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != bg:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                                comp.append((nr, nc))
                groups.append(comp)

    # Find template group (has all non-bg colors)
    template_group = None
    for g in groups:
        colors = set(grid[r][c] for r, c in g)
        if colors == all_colors:
            template_group = g
            break

    if template_group is None:
        return out

    template_positions = set((r, c) for r, c in template_group)
    min_r = min(r for r, c in template_group)
    min_c = min(c for r, c in template_group)
    template_cells = [(r - min_r, c - min_c, grid[r][c]) for r, c in template_group]

    # Generate all 8 transformations
    def all_transforms(cells):
        results = []
        for rot in range(4):
            for flip in [False, True]:
                t = []
                for dr, dc, col in cells:
                    r, c = dr, dc
                    for _ in range(rot):
                        r, c = c, -r
                    if flip:
                        c = -c
                    t.append((r, c, col))
                mr = min(r for r, c, _ in t)
                mc = min(c for r, c, _ in t)
                t = [(r - mr, c - mc, col) for r, c, col in t]
                results.append(t)
        return results

    transforms = all_transforms(template_cells)

    # Find all valid placements
    placements = []
    for ti, trans in enumerate(transforms):
        max_dr = max(r for r, c, _ in trans)
        max_dc = max(c for r, c, _ in trans)
        for start_r in range(0 - max_dr, h):
            for start_c in range(0 - max_dc, w):
                matches = 0
                missing = 0
                conflicts = 0
                match_cells = []
                missing_cells = []

                for dr, dc, col in trans:
                    r, c = start_r + dr, start_c + dc
                    if r < 0 or r >= h or c < 0 or c >= w:
                        conflicts += 1
                        break
                    if (r, c) in template_positions:
                        conflicts += 1
                        break
                    if grid[r][c] == col:
                        matches += 1
                        match_cells.append((r, c, col))
                    elif grid[r][c] == bg:
                        missing += 1
                        missing_cells.append((r, c, col))
                    else:
                        conflicts += 1
                        break

                if conflicts == 0 and matches >= 2:
                    placements.append({
                        'matches': matches,
                        'match_cells': match_cells,
                        'missing_cells': missing_cells
                    })

    # Sort by matches descending
    placements.sort(key=lambda p: -p['matches'])

    # Greedy selection: cover all non-template non-bg cells
    non_template_cells = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in template_positions:
                non_template_cells.add((r, c))

    covered = set()
    for p in placements:
        match_pos = set((r, c) for r, c, _ in p['match_cells'])
        if match_pos.issubset(non_template_cells) and not match_pos.issubset(covered):
            new_covered = match_pos - covered
            if new_covered:
                covered |= match_pos
                for r, c, col in p['missing_cells']:
                    out[r][c] = col

    return out
