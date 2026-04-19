import copy
from collections import Counter


def transform(input_grid):
    grid = copy.deepcopy(input_grid)
    R = len(grid)
    C = len(grid[0])

    # Determine background color (most common)
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components (8-connectivity)
    visited = set()
    components = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg and (r, c) not in visited:
                comp = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc, grid[cr][cc]))
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited and grid[nr][nc] != bg:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                components.append(comp)

    templates = [c for c in components if len(set(v for _, _, v in c)) > 1]
    single_color_groups = [c for c in components if len(set(v for _, _, v in c)) == 1 and len(c) > 1]
    standalone_pixels = [c for c in components if len(c) == 1]

    template_colors = set()
    for t in templates:
        template_colors.update(v for _, _, v in t)

    # Normalize each template
    template_shapes = []
    for t in templates:
        min_r = min(r for r, c, v in t)
        min_c = min(c for r, c, v in t)
        cells = [(r - min_r, c - min_c, v) for r, c, v in t]
        template_shapes.append(cells)

    output = copy.deepcopy(grid)

    # Step 1: Stamp templates at standalone pixels
    for ti, t in enumerate(templates):
        shape = template_shapes[ti]
        colors_in_template = set(v for _, _, v in shape)

        for s in standalone_pixels:
            sr, sc, sv = s[0]
            if sv not in colors_in_template:
                continue

            positions_of_color = [(dr, dc) for dr, dc, v in shape if v == sv]

            best_pos = None
            best_overlap = -1
            for pr, pc in positions_of_color:
                origin_r, origin_c = sr - pr, sc - pc
                overlap = 0
                conflict = False
                for dr, dc, v in shape:
                    tr, tc = origin_r + dr, origin_c + dc
                    if 0 <= tr < R and 0 <= tc < C:
                        if grid[tr][tc] == v:
                            overlap += 1
                        elif grid[tr][tc] != bg:
                            conflict = True
                            break
                if not conflict and overlap > best_overlap:
                    best_overlap = overlap
                    best_pos = (pr, pc)

            if best_pos is not None:
                pr, pc = best_pos
                origin_r, origin_c = sr - pr, sc - pc
                for dr, dc, v in shape:
                    tr, tc = origin_r + dr, origin_c + dc
                    if 0 <= tr < R and 0 <= tc < C:
                        output[tr][tc] = v

    # Step 2: Complete partial single-color groups that match a template
    for g in single_color_groups:
        gcolor = g[0][2]
        if gcolor not in template_colors:
            continue

        g_cells = sorted([(r, c) for r, c, _ in g])

        for ti, shape in enumerate(template_shapes):
            t_positions = [(dr, dc) for dr, dc, v in shape if v == gcolor]
            if len(t_positions) < len(g_cells):
                continue

            matched = False
            for anchor_idx in range(len(g_cells)):
                for t_idx in range(len(t_positions)):
                    ar, ac = g_cells[anchor_idx]
                    tr, tc = t_positions[t_idx]
                    origin_r, origin_c = ar - tr, ac - tc

                    all_match = True
                    for gr, gc in g_cells:
                        rel_r, rel_c = gr - origin_r, gc - origin_c
                        if (rel_r, rel_c) not in t_positions:
                            all_match = False
                            break

                    if all_match:
                        conflict = False
                        for dr, dc, v in shape:
                            nr, nc = origin_r + dr, origin_c + dc
                            if 0 <= nr < R and 0 <= nc < C:
                                if output[nr][nc] != bg and output[nr][nc] != v:
                                    conflict = True
                                    break
                        if not conflict:
                            for dr, dc, v in shape:
                                nr, nc = origin_r + dr, origin_c + dc
                                if 0 <= nr < R and 0 <= nc < C:
                                    if output[nr][nc] == bg:
                                        output[nr][nc] = v
                            matched = True
                            break
                if matched:
                    break
            if matched:
                break

    # Step 3: Handle orphan single-color groups (color not in any template)
    orphans = [g for g in single_color_groups if g[0][2] not in template_colors]
    if not orphans:
        return output

    orphan_color = orphans[0][0][2]
    ref_orphan = orphans[0]
    min_r_o = min(r for r, c, _ in ref_orphan)
    min_c_o = min(c for r, c, _ in ref_orphan)
    orphan_shape = [(r - min_r_o, c - min_c_o) for r, c, _ in ref_orphan]

    orphan_rows = set()
    for g in orphans:
        rows = set(r for r, c, _ in g)
        orphan_rows.update(rows)

    template_row_ranges = []
    for t in templates:
        t_min_r = min(r for r, c, v in t)
        t_max_r = max(r for r, c, v in t)
        template_row_ranges.append((t_min_r, t_max_r))

    # Build per-template color sets for checking multi-template rows
    template_color_sets = []
    for t in templates:
        template_color_sets.append(set(v for _, _, v in t))

    # Step 3a: Midpoint rule (runs FIRST)
    # Only fires on rows with standalone pixels from >=2 different templates
    standalone_by_row = {}
    for s in standalone_pixels:
        sr, sc, sv = s[0]
        if sr not in standalone_by_row:
            standalone_by_row[sr] = []
        standalone_by_row[sr].append((sc, sv))

    for row, pixels in standalone_by_row.items():
        if row in orphan_rows:
            continue

        # Check how many different templates are represented
        represented_templates = set()
        for _, sv in pixels:
            for ti, cs in enumerate(template_color_sets):
                if sv in cs:
                    represented_templates.add(ti)

        if len(represented_templates) < 2:
            continue

        by_color = {}
        for col, val in pixels:
            if val not in by_color:
                by_color[val] = []
            by_color[val].append(col)

        for color, cols in by_color.items():
            if len(cols) == 2:
                midpoint = (cols[0] + cols[1]) // 2
                can_place = True
                for dr, dc in orphan_shape:
                    nr, nc = row + dr, midpoint + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        if output[nr][nc] != bg and output[nr][nc] != orphan_color:
                            can_place = False
                            break
                    else:
                        can_place = False
                        break

                if can_place:
                    for dr, dc in orphan_shape:
                        nr, nc = row + dr, midpoint + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            output[nr][nc] = orphan_color
                    orphan_rows.add(row)
                    break

    # Step 3b: Propagation (runs SECOND, skips rows with existing orphans)
    for g in orphans:
        g_rows = set(r for r, c, _ in g)
        g_cols = sorted(set(c for r, c, _ in g))
        g_min_c = g_cols[0]

        shared_template = None
        for t_idx, t in enumerate(templates):
            t_rows = set(r for r, c, v in t)
            if g_rows & t_rows:
                shared_template = t_idx
                break

        if shared_template is None:
            continue

        t_min_r, t_max_r = template_row_ranges[shared_template]
        template_center = (t_min_r + t_max_r) / 2
        grid_center = (R - 1) / 2

        if template_center <= grid_center:
            start_row = R - 1
            step = -2
            stop_row = t_max_r + 1
        else:
            start_row = 0
            step = 2
            stop_row = t_min_r - 1

        r = start_row
        while (step > 0 and r <= stop_row) or (step < 0 and r >= stop_row):
            if r not in orphan_rows:
                can_place = True
                for dr, dc in orphan_shape:
                    nr, nc = r + dr, g_min_c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        if output[nr][nc] != bg and output[nr][nc] != orphan_color:
                            can_place = False
                            break
                    else:
                        can_place = False
                        break

                if can_place:
                    for dr, dc in orphan_shape:
                        nr, nc = r + dr, g_min_c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            output[nr][nc] = orphan_color
                    orphan_rows.add(r)
            r += step

    return output