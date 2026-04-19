"""
61dd76ce solver: Grid-cell transformation
- Grid lines = rows/cols of all-same-color (grid_color, either 5 or 9)
- Each cell between grid lines classified:
  1-marker → 3, none → 8 or fill, extra-grid → 8 or fill/grid
- Standard case (fill=5,grid=9): extra→8, none→5, mystery outermost→9
- Inverted case (grid=5): extra→fill or grid_color based on marker matching
"""
from collections import Counter

def transform(grid):
    H, W = len(grid), len(grid[0])
    
    # Detect grid_color (try 9 first, then 5)
    grid_color = 9
    h_lines = [r for r in range(H) if all(grid[r][c] == grid_color for c in range(W))]
    v_lines = [c for c in range(W) if all(grid[r][c] == grid_color for r in range(H))]
    if not h_lines and not v_lines:
        grid_color = 5
        h_lines = [r for r in range(H) if all(grid[r][c] == grid_color for c in range(W))]
        v_lines = [c for c in range(W) if all(grid[r][c] == grid_color for r in range(H))]

    def build_groups(lines, size):
        groups, prev = [], -1
        for l in sorted(lines) + [size]:
            if l > prev + 1:
                groups.append((prev + 1, l - 1))
            prev = l
        return groups

    row_groups = build_groups(h_lines, H)
    col_groups = build_groups(v_lines, W)
    n_rg = len(row_groups)

    # Determine fill_color from first cell
    if row_groups and col_groups:
        r0, r1 = row_groups[0]
        c0, c1 = col_groups[0]
        cell_vals = [grid[r][c] for r in range(r0, r1+1) for c in range(c0, c1+1)]
        ctr = Counter(v for v in cell_vals if v != 1)
        fill_color = ctr.most_common(1)[0][0] if ctr else 5
    else:
        fill_color = 5

    # Classify each cell
    cell_type = {}
    cell_extra_rel = {}  # (ri,ci) -> (rel_r, rel_c) of extra pixel
    cell_one_rel = {}    # (ri,ci) -> list of (rel_r, rel_c) of 1-markers
    
    for ri, (r0, r1) in enumerate(row_groups):
        for ci, (c0, c1) in enumerate(col_groups):
            has_1 = any(grid[r][c] == 1 for r in range(r0, r1+1) for c in range(c0, c1+1))
            has_extra = any(grid[r][c] == grid_color for r in range(r0, r1+1) for c in range(c0, c1+1))
            if has_1:
                cell_type[(ri, ci)] = '1-marker'
                ones = [(r-r0, c-c0) for r in range(r0, r1+1) for c in range(c0, c1+1) if grid[r][c] == 1]
                cell_one_rel[(ri, ci)] = ones
            elif has_extra:
                cell_type[(ri, ci)] = 'extra'
                for r in range(r0, r1+1):
                    for c in range(c0, c1+1):
                        if grid[r][c] == grid_color:
                            cell_extra_rel[(ri, ci)] = (r - r0, c - c0)
                            break
                    if (ri, ci) in cell_extra_rel:
                        break
            else:
                cell_type[(ri, ci)] = 'none'

    # Collect all marker relative positions with their cell coords
    all_marker_positions = set()
    marker_pos_to_cells = {}
    for (ri, ci), ones in cell_one_rel.items():
        for pos in ones:
            all_marker_positions.add(pos)
            marker_pos_to_cells.setdefault(pos, []).append((ri, ci))

    # Mystery 9 cells (standard case: fill=5, grid=9)
    mystery_9_cells = set()
    if fill_color == 5:
        for ri, (r0, r1) in enumerate(row_groups):
            for ci, (c0, c1) in enumerate(col_groups):
                if cell_type.get((ri, ci)) != 'extra':
                    continue
                er, ec = cell_extra_rel.get((ri, ci), (0, 0))
                abs_r = r0 + er
                at_outer_edge = False
                if ri == 0 and abs_r == 0:
                    at_outer_edge = True
                if ri == n_rg - 1 and abs_r == H - 1:
                    at_outer_edge = True
                if at_outer_edge:
                    opp_ri = n_rg - 1 - ri
                    if opp_ri != ri:
                        mystery_9_cells.add((opp_ri, ci))

    def get_output_val(ri, ci):
        ct = cell_type.get((ri, ci), 'none')
        if ct == '1-marker':
            return 3
        if fill_color == 5:
            if ct == 'extra':
                return 8
            if (ri, ci) in mystery_9_cells:
                return 9
            return fill_color
        else:
            # Inverted case (grid=5)
            if ct == 'extra':
                er, ec = cell_extra_rel.get((ri, ci), (0, 0))
                r0, r1 = row_groups[ri]
                ch = r1 - r0 + 1
                hrev = (ch - 1 - er, ec)
                # Rule 1: H-reflection matches a marker position
                if hrev in all_marker_positions:
                    return grid_color
                # Rule 2: Original matches marker in different col group
                if (er, ec) in all_marker_positions:
                    for mri, mci in marker_pos_to_cells.get((er, ec), []):
                        if mci != ci:
                            return grid_color
                return fill_color
            return 8

    out = [row[:] for row in grid]
    for ri, (r0, r1) in enumerate(row_groups):
        for ci, (c0, c1) in enumerate(col_groups):
            val = get_output_val(ri, ci)
            for r in range(r0, r1+1):
                for c in range(c0, c1+1):
                    out[r][c] = val
    return out
