"""
Solver for ARC puzzle 6f116a3b.

Rule:
- Background is the most common color.
- A "frame" color forms large connected shapes (templates) containing colored anchor points.
- Each template has: frame cells (frame color) + anchor cells (other colors).
- Scattered anchor groups exist elsewhere with the same color pattern as template anchors.
- For each scattered group, the template's frame is stamped around it using the D4 symmetry
  transformation that maps the template's anchor pattern to the scattered group's anchor pattern.
- Original templates are removed (set to background).
- If no frame color exists, groups that are exact-orientation duplicates of other groups are removed.
"""

from collections import Counter


def transform(grid):
    R, C = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    bg = Counter(flat).most_common(1)[0][0]

    # Collect non-bg cells
    non_bg = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != bg:
                non_bg[(r, c)] = grid[r][c]

    # 8-connected components
    def flood_8(all_pos):
        components = []
        visited = set()
        for s in sorted(all_pos):
            if s in visited:
                continue
            comp = set()
            stack = [s]
            while stack:
                p = stack.pop()
                if p in comp:
                    continue
                comp.add(p)
                pr, pc = p
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nb = (pr + dr, pc + dc)
                        if nb in all_pos and nb not in comp:
                            stack.append(nb)
            visited |= comp
            components.append(comp)
        return components

    comps = flood_8(set(non_bg.keys()))

    # Find frame color: dominant color in large components
    frame_color = None
    for comp in comps:
        if len(comp) < 5:
            continue
        freq = Counter(non_bg[p] for p in comp)
        dom_color, dom_count = freq.most_common(1)[0]
        if dom_count / len(comp) > 0.5:
            frame_color = dom_color
            break

    # D4 symmetry transforms
    d4 = [
        lambda r, c: (r, c),       # identity
        lambda r, c: (c, -r),      # 90° CW
        lambda r, c: (-r, -c),     # 180°
        lambda r, c: (-c, r),      # 270° CW
        lambda r, c: (-r, c),      # flip x
        lambda r, c: (r, -c),      # flip y
        lambda r, c: (c, r),       # transpose
        lambda r, c: (-c, -r),     # anti-transpose
    ]

    result = [row[:] for row in grid]

    if frame_color is not None:
        return _solve_with_frame(result, comps, non_bg, bg, frame_color, d4, R, C)
    else:
        return _solve_no_frame(result, comps, non_bg, bg, d4, R, C)


def _solve_with_frame(result, comps, non_bg, bg, frame_color, d4, R, C):
    # Separate templates and scattered
    templates = []
    scattered = {}
    for comp in comps:
        if any(non_bg[p] == frame_color for p in comp):
            templates.append(comp)
        else:
            for p in comp:
                scattered[p] = non_bg[p]

    # Extract template data
    template_data = []
    for comp in templates:
        frame_cells = [(r, c) for (r, c) in comp if non_bg[(r, c)] == frame_color]
        anchor_cells = [(r, c, non_bg[(r, c)]) for (r, c) in comp if non_bg[(r, c)] != frame_color]
        template_data.append({
            'frame': frame_cells,
            'anchors': anchor_cells,
            'component': comp,
        })

    # Remove all template cells from result
    for td in template_data:
        for (r, c) in td['component']:
            result[r][c] = bg

    # Anchor colors
    anchor_colors = sorted(set(non_bg[p] for p in scattered))

    # For matching, pick each anchor color as potential reference
    # For each scattered cell of that color, try matching to each template
    used_scattered = set()

    # Try each anchor color as reference
    for ref_color in anchor_colors:
        ref_cells = [(r, c) for (r, c), v in scattered.items() if v == ref_color]
        for ref_r, ref_c in ref_cells:
            if (ref_r, ref_c) in used_scattered:
                continue

            matched = False
            for td in template_data:
                # Find anchors of ref_color in this template
                t_refs = [(r, c) for r, c, v in td['anchors'] if v == ref_color]
                other_anchors = [(r, c, v) for r, c, v in td['anchors'] if v != ref_color]

                for t_ref_r, t_ref_c in t_refs:
                    for F in d4:
                        # Check if all other anchors match
                        match = True
                        matched_cells = [(ref_r, ref_c)]

                        for oa_r, oa_c, oa_v in other_anchors:
                            dr, dc = oa_r - t_ref_r, oa_c - t_ref_c
                            tdr, tdc = F(dr, dc)
                            exp_r, exp_c = ref_r + tdr, ref_c + tdc

                            if ((exp_r, exp_c) in scattered and
                                    scattered[(exp_r, exp_c)] == oa_v and
                                    (exp_r, exp_c) not in used_scattered):
                                matched_cells.append((exp_r, exp_c))
                            else:
                                match = False
                                break

                        if match:
                            # Stamp frame
                            for f_r, f_c in td['frame']:
                                dr, dc = f_r - t_ref_r, f_c - t_ref_c
                                tdr, tdc = F(dr, dc)
                                sr, sc = ref_r + tdr, ref_c + tdc
                                if 0 <= sr < R and 0 <= sc < C:
                                    result[sr][sc] = frame_color

                            for mc in matched_cells:
                                used_scattered.add(mc)
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break

    return result


def _solve_no_frame(result, comps, non_bg, bg, d4, R, C):
    # No frame color: each D4 orbit should have exactly 4 groups.
    # Remove extras to get there.

    # Compute signature and canonical form for each group
    groups = []
    for comp in comps:
        cells = sorted(comp)
        base_r, base_c = cells[0]
        sig = tuple(sorted((non_bg[(r, c)], r - base_r, c - base_c) for r, c in cells))

        # Canonical D4 form
        best_canon = None
        for F in d4:
            transformed = tuple(sorted((color, F(dr, dc)[0], F(dr, dc)[1])
                                       for color, dr, dc in sig))
            min_r = min(t[1] for t in transformed)
            min_c = min(t[2] for t in transformed)
            normalized = tuple(sorted((color, r - min_r, c - min_c)
                                      for color, r, c in transformed))
            if best_canon is None or normalized < best_canon:
                best_canon = normalized

        centroid_r = sum(r for r, c in cells) / len(cells)
        centroid_c = sum(c for r, c in cells) / len(cells)
        groups.append({
            'comp': comp, 'sig': sig, 'canon': best_canon,
            'centroid': (centroid_r, centroid_c),
        })

    # Group by D4 orbit
    orbits = {}
    for i, g in enumerate(groups):
        orbits.setdefault(g['canon'], []).append(i)

    to_remove = set()
    target_per_orbit = 4

    for canon, indices in orbits.items():
        while len(indices) > target_per_orbit:
            # Find duplicate orientations
            sig_groups = {}
            for i in indices:
                sig_groups.setdefault(groups[i]['sig'], []).append(i)

            dup_indices = []
            for sig, members in sig_groups.items():
                if len(members) > 1:
                    dup_indices.extend(members)

            if dup_indices:
                # Remove the duplicate with the highest centroid row
                worst = max(dup_indices, key=lambda i: groups[i]['centroid'][0])
                to_remove |= groups[worst]['comp']
                indices.remove(worst)
            else:
                # No duplicates: remove the most central (lowest sum of distances)
                def sum_dist(idx):
                    cr, cc = groups[idx]['centroid']
                    return sum(((cr - groups[j]['centroid'][0]) ** 2 +
                                (cc - groups[j]['centroid'][1]) ** 2) ** 0.5
                               for j in indices if j != idx)
                most_central = min(indices, key=sum_dist)
                to_remove |= groups[most_central]['comp']
                indices.remove(most_central)

    for r, c in to_remove:
        result[r][c] = bg

    return result
