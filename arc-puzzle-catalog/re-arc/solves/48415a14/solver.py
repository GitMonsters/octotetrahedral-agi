"""Solver for ARC puzzle 48415a14.

Rule: Grid of cells separated by grid-colored lines.
- Find the "template" pattern (connected non-bg cells including grid-colored cells in cell positions)
- Find "markers" (isolated grid-colored cells in cell positions)
- For each marker, stamp a copy of the template, anchored at the marker position
  matching the first grid-colored cell in the template (reading order)
- If no markers, tile the pattern using brick-wall tiling vectors derived from bounding box
"""
from collections import Counter, deque


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    grid = [list(row) for row in input_grid]
    rows, cols = len(grid), len(grid[0])

    # Find separator rows and cols
    h_seps = [r for r in range(rows) if len(set(grid[r])) == 1]
    v_seps = [c for c in range(cols) if len(set(grid[r][c] for r in range(rows))) == 1]
    grid_color = grid[h_seps[0]][0]

    # Cell ranges
    def ranges_from_seps(seps, total):
        res, prev = [], 0
        for s in seps:
            if s > prev:
                res.append((prev, s))
            prev = s + 1
        if prev < total:
            res.append((prev, total))
        return res

    cell_rows = ranges_from_seps(h_seps, rows)
    cell_cols = ranges_from_seps(v_seps, cols)
    nr, nc = len(cell_rows), len(cell_cols)

    # Extract cell grid
    cell_grid = []
    for cr_s, cr_e in cell_rows:
        row = []
        for cc_s, cc_e in cell_cols:
            row.append(grid[cr_s][cc_s])
        cell_grid.append(row)

    # Background color
    c = Counter(v for r in cell_grid for v in r)
    bg = c.most_common(1)[0][0]

    # Non-bg cells
    non_bg = {}
    for r in range(nr):
        for col in range(nc):
            if cell_grid[r][col] != bg:
                non_bg[(r, col)] = cell_grid[r][col]

    # Connected components (8-connectivity)
    def find_components(cells):
        remaining = set(cells.keys())
        components = []
        while remaining:
            start = next(iter(remaining))
            comp = {}
            queue = deque([start])
            while queue:
                pos = queue.popleft()
                if pos in remaining:
                    remaining.remove(pos)
                    comp[pos] = cells[pos]
                    r, col = pos
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nb = (r + dr, col + dc)
                            if nb in remaining:
                                queue.append(nb)
            components.append(comp)
        return components

    comps = find_components(non_bg)

    # Separate template (has non-grid colors) from markers (grid-colored only)
    template = None
    markers = []
    for comp in comps:
        has_non_grid = any(v != grid_color for v in comp.values())
        if has_non_grid:
            if template is None or len(comp) > len(template):
                if template is not None:
                    # Previous template was smaller, treat as markers
                    for pos, val in template.items():
                        if val == grid_color:
                            markers.append(pos)
                template = comp
            else:
                for pos, val in comp.items():
                    if val == grid_color:
                        markers.append(pos)
        else:
            for pos in comp:
                markers.append(pos)

    if template is None:
        return grid  # No pattern found

    # Template bounding box
    t_min_r = min(r for r, _ in template)
    t_max_r = max(r for r, _ in template)
    t_min_c = min(c for _, c in template)
    t_max_c = max(c for _, c in template)
    bb_h = t_max_r - t_min_r + 1
    bb_w = t_max_c - t_min_c + 1

    # Template as relative offsets from bounding box TL
    template_rel = {}
    for (r, col), val in template.items():
        template_rel[(r - t_min_r, col - t_min_c)] = val

    # Output grid (copy of input)
    out = [list(row) for row in input_grid]

    def stamp_template(base_r, base_c):
        """Stamp template at cell position (base_r, base_c) as the BB top-left"""
        for (dr, dc), val in template_rel.items():
            cr, cc = base_r + dr, base_c + dc
            if 0 <= cr < nr and 0 <= cc < nc:
                # Write to pixel grid
                pr_s, pr_e = cell_rows[cr]
                pc_s, pc_e = cell_cols[cc]
                for pr in range(pr_s, pr_e):
                    for pc in range(pc_s, pc_e):
                        out[pr][pc] = val

    if markers:
        # Find anchor: first grid-colored cell in template (reading order)
        grid_cells_in_template = sorted(
            [(r, c) for (r, c), v in template.items() if v == grid_color]
        )
        if grid_cells_in_template:
            anchor = grid_cells_in_template[0]
            anchor_rel = (anchor[0] - t_min_r, anchor[1] - t_min_c)
        else:
            anchor_rel = (0, 0)

        for m_r, m_c in markers:
            base_r = m_r - anchor_rel[0]
            base_c = m_c - anchor_rel[1]
            stamp_template(base_r, base_c)
    else:
        # No markers: tile using brick-wall pattern
        # Two column series per row block:
        #   Series A (brick): base_c + block_idx * BB_w  (shifts per row block)
        #   Series B (fixed): base_c + BB_h + BB_w  (same for all row blocks)
        # Additional copies at multiples of (BB_h + BB_w) if grid is wide enough
        base_r, base_c = t_min_r, t_min_c
        v_col = bb_h + bb_w  # horizontal period

        positions = set()
        for block_idx in range(-20, 20):
            copy_r = base_r - block_idx * bb_h
            # Series A: brick-shifted columns
            brick_col_start = base_c + block_idx * bb_w
            for j in range(-10, 10):
                copy_c = brick_col_start + j * v_col
                if fits(copy_r, copy_c, nr, nc, template_rel):
                    positions.add((copy_r, copy_c))
            # Series B: unshifted columns (skip j=0, covered by Series A for base block)
            for j in range(1, 10):
                copy_c = base_c + j * v_col
                if fits(copy_r, copy_c, nr, nc, template_rel):
                    positions.add((copy_r, copy_c))

        for r, c in positions:
            if (r, c) != (base_r, base_c):
                stamp_template(r, c)

    return out


def fits(base_r, base_c, nr, nc, template_rel):
    """Check if template fits at this base position"""
    for dr, dc in template_rel:
        cr, cc = base_r + dr, base_c + dc
        if not (0 <= cr < nr and 0 <= cc < nc):
            return False
    return True
