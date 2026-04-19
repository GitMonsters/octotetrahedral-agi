import json
from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    flat = [c for row in grid for c in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    other_colors = [c for c, _ in counts.most_common() if c != bg]
    
    if len(other_colors) >= 2:
        return _transform_3color(grid, rows, cols, bg, other_colors, counts)
    elif len(other_colors) == 1:
        return _transform_2color(grid, rows, cols, bg, other_colors[0])
    else:
        return [row[:] for row in grid]


def _transform_3color(grid, rows, cols, bg, other_colors, counts):
    border_color = min(other_colors, key=lambda c: counts[c])
    interior_color = max(other_colors, key=lambda c: counts[c])
    
    border_cells = [(r, c) for r in range(rows) for c in range(cols)
                    if grid[r][c] == border_color]
    min_r = min(r for r, c in border_cells)
    max_r = max(r for r, c in border_cells)
    min_c = min(c for r, c in border_cells)
    max_c = max(c for r, c in border_cells)
    
    template_interior = set()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == interior_color:
                template_interior.add((r, c))
    
    ti_min_r = min(r for r, c in template_interior)
    ti_min_c = min(c for r, c in template_interior)
    template_interior_rel = frozenset(
        (r - ti_min_r, c - ti_min_c) for r, c in template_interior
    )
    
    all_orientations = _get_all_orientations(template_interior_rel)
    components = _find_components(grid, rows, cols, interior_color)
    
    output = [row[:] for row in grid]
    
    for comp in components:
        comp_set = set(comp)
        if comp_set == template_interior:
            continue
        if len(comp) != len(template_interior):
            continue
        
        comp_min_r = min(r for r, c in comp)
        comp_min_c = min(c for r, c in comp)
        comp_rel = frozenset((r - comp_min_r, c - comp_min_c) for r, c in comp)
        
        if comp_rel not in all_orientations:
            continue
        
        comp_max_r = max(r for r, c in comp)
        comp_max_c = max(c for r, c in comp)
        br_min_r = comp_min_r - 1
        br_max_r = comp_max_r + 1
        br_min_c = comp_min_c - 1
        br_max_c = comp_max_c + 1
        
        conflict = False
        for r in range(max(0, br_min_r), min(rows, br_max_r + 1)):
            for c in range(max(0, br_min_c), min(cols, br_max_c + 1)):
                if (r, c) not in comp_set and grid[r][c] != bg:
                    conflict = True
                    break
            if conflict:
                break
        
        if conflict:
            continue
        
        for r in range(max(0, br_min_r), min(rows, br_max_r + 1)):
            for c in range(max(0, br_min_c), min(cols, br_max_c + 1)):
                if (r, c) not in comp_set:
                    output[r][c] = border_color
    
    return output


def _transform_2color(grid, rows, cols, bg, non_bg):
    frame = _find_frame(grid, rows, cols, bg, non_bg)
    if frame:
        return _transform_2color_frame(grid, rows, cols, bg, non_bg, frame)
    else:
        return _transform_2color_noframe(grid, rows, cols, bg, non_bg)


def _find_frame(grid, rows, cols, bg, non_bg):
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1 + 2, rows):
                for c2 in range(c1 + 2, cols):
                    ok = True
                    for c in range(c1, c2 + 1):
                        if grid[r1][c] == bg or grid[r2][c] == bg:
                            ok = False
                            break
                    if not ok:
                        continue
                    for r in range(r1 + 1, r2):
                        if grid[r][c1] == bg or grid[r][c2] == bg:
                            ok = False
                            break
                    if not ok:
                        continue
                    for r in range(r1 + 1, r2):
                        for c in range(c1 + 1, c2):
                            if grid[r][c] == bg:
                                return (r1, c1, r2, c2)
    return None


def _transform_2color_frame(grid, rows, cols, bg, non_bg, frame):
    r1, c1, r2, c2 = frame
    h = r2 - r1 + 1
    w = c2 - c1 + 1
    
    pattern = []
    for r in range(r1, r2 + 1):
        row = []
        for c in range(c1, c2 + 1):
            row.append(grid[r][c])
        pattern.append(row)
    
    holes = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if pattern[r][c] == bg:
                holes.append((r, c))
    
    if not holes:
        return [row[:] for row in grid]
    
    hole_rows = sorted(set(hr for hr, hc in holes))
    hole_cols = sorted(set(hc for hr, hc in holes))
    max_hr = max(hole_rows)
    hc = hole_cols[0]
    nh = len(holes)
    
    offsets = [
        (-h, -(w + hc)),
        (-nh, w),
        (h - max_hr - 1, -2 * w),
        (h - max_hr - 1, w),
    ]
    
    output = [row[:] for row in grid]
    placed_regions = [(r1, c1, r2, c2)]
    
    for dr, dc in offsets:
        nr1 = r1 + dr
        nc1 = c1 + dc
        nr2 = nr1 + h - 1
        nc2 = nc1 + w - 1
        
        vis_r1 = max(0, nr1)
        vis_r2 = min(rows - 1, nr2)
        vis_c1 = max(0, nc1)
        vis_c2 = min(cols - 1, nc2)
        
        if vis_r1 > vis_r2 or vis_c1 > vis_c2:
            continue
        
        overlap = False
        for pr1, pc1, pr2, pc2 in placed_regions:
            if nr1 <= pr2 and nr2 >= pr1 and nc1 <= pc2 and nc2 >= pc1:
                overlap = True
                break
        if overlap:
            continue
        
        conflict = False
        for r in range(vis_r1, vis_r2 + 1):
            for c in range(vis_c1, vis_c2 + 1):
                if grid[r][c] != bg:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        
        for r in range(vis_r1, vis_r2 + 1):
            for c in range(vis_c1, vis_c2 + 1):
                pr, pc = r - nr1, c - nc1
                output[r][c] = pattern[pr][pc]
        
        placed_regions.append((nr1, nc1, nr2, nc2))
    
    return output


def _transform_2color_noframe(grid, rows, cols, bg, non_bg):
    """Handle 2-color case without a frame: border all matching-shape components."""
    components = _find_components(grid, rows, cols, non_bg)
    
    if not components:
        return [row[:] for row in grid]
    
    # Group components by shape (considering all 8 orientations)
    shape_to_comps = {}
    comp_to_canonical = {}
    
    for i, comp in enumerate(components):
        shape = _normalize_shape(comp)
        oris = _get_all_orientations(shape)
        canonical = min(oris, key=lambda s: tuple(sorted(s)))
        
        if canonical not in shape_to_comps:
            shape_to_comps[canonical] = []
        shape_to_comps[canonical].append(i)
        comp_to_canonical[i] = canonical
    
    # Find shapes with 2+ matching components
    matching_shapes = {s for s, idxs in shape_to_comps.items() if len(idxs) >= 2}
    
    output = [row[:] for row in grid]
    
    for shape in matching_shapes:
        for comp_idx in shape_to_comps[shape]:
            comp = components[comp_idx]
            comp_set = set(comp)
            
            comp_min_r = min(r for r, c in comp)
            comp_max_r = max(r for r, c in comp)
            comp_min_c = min(c for r, c in comp)
            comp_max_c = max(c for r, c in comp)
            
            br_min_r = comp_min_r - 1
            br_max_r = comp_max_r + 1
            br_min_c = comp_min_c - 1
            br_max_c = comp_max_c + 1
            
            # Check for conflicts (non-bg, non-comp cells in border region)
            conflict = False
            for r in range(max(0, br_min_r), min(rows, br_max_r + 1)):
                for c in range(max(0, br_min_c), min(cols, br_max_c + 1)):
                    if (r, c) not in comp_set and grid[r][c] != bg:
                        conflict = True
                        break
                if conflict:
                    break
            
            if conflict:
                continue
            
            for r in range(max(0, br_min_r), min(rows, br_max_r + 1)):
                for c in range(max(0, br_min_c), min(cols, br_max_c + 1)):
                    if (r, c) not in comp_set:
                        output[r][c] = non_bg
    
    return output


def _normalize_shape(comp):
    min_r = min(r for r, c in comp)
    min_c = min(c for r, c in comp)
    return frozenset((r - min_r, c - min_c) for r, c in comp)


def _get_all_orientations(positions):
    def normalize(pos_list):
        min_r = min(r for r, c in pos_list)
        min_c = min(c for r, c in pos_list)
        return frozenset((r - min_r, c - min_c) for r, c in pos_list)
    
    orientations = set()
    pos = list(positions)
    
    for _ in range(4):
        orientations.add(normalize(pos))
        max_c = max(c for r, c in pos)
        flipped = [(r, max_c - c) for r, c in pos]
        orientations.add(normalize(flipped))
        max_r = max(r for r, c in pos)
        pos = [(c, max_r - r) for r, c in pos]
    
    return orientations


def _find_components(grid, rows, cols, color):
    visited = set()
    components = []
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and (r, c) not in visited:
                comp = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols
                                and (nr, nc) not in visited
                                and grid[nr][nc] == color):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)
    return components
