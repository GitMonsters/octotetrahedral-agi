import math
from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    bg = Counter(v for r in input_grid for v in r).most_common(1)[0][0]
    
    by_color = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                by_color.setdefault(input_grid[r][c], []).append((r, c))
    
    if not by_color:
        return input_grid
    
    colors = list(by_color.keys())
    
    if len(colors) == 1:
        color = colors[0]
        all_cells = by_color[color]
        cell_set = set(all_cells)
        new_color = color
        
        # Find marker: the cell whose removal most reduces bbox area
        full_bbox_area = (max(r for r,c in all_cells) - min(r for r,c in all_cells) + 1) * \
                         (max(c for r,c in all_cells) - min(c for r,c in all_cells) + 1)
        
        best_marker = None
        best_area = full_bbox_area
        for cr, cc in all_cells:
            remaining = [(r,c) for r,c in all_cells if (r,c) != (cr,cc)]
            if not remaining:
                continue
            area = (max(r for r,c in remaining) - min(r for r,c in remaining) + 1) * \
                   (max(c for r,c in remaining) - min(c for r,c in remaining) + 1)
            if area < best_area:
                best_area = area
                best_marker = (cr, cc)
        
        if best_marker:
            marker_cells = {best_marker}
            shape_cells = [x for x in all_cells if x != best_marker]
        else:
            return input_grid
    elif len(colors) == 2:
        c1, c2 = colors
        if len(by_color[c1]) > len(by_color[c2]):
            shape_color, marker_color = c1, c2
        else:
            shape_color, marker_color = c2, c1
        shape_cells = list(by_color[shape_color])
        marker_cells = set(by_color[marker_color])
        new_color = marker_color
    else:
        return input_grid
    
    if not marker_cells or not shape_cells:
        return input_grid
    
    shape_set = set(shape_cells)
    
    # Compute reflection center (average of midpoints)
    midpoints = []
    for mr, mc in marker_cells:
        best_dist = float('inf')
        best_cell = None
        for sr, sc in shape_cells:
            d = max(abs(sr - mr), abs(sc - mc))
            if d < best_dist:
                best_dist = d
                best_cell = (sr, sc)
        midpoints.append(((mr + best_cell[0]) / 2, (mc + best_cell[1]) / 2))
    
    center_r = sum(m[0] for m in midpoints) / len(midpoints)
    center_c = sum(m[1] for m in midpoints) / len(midpoints)
    
    # Compute stamp shift
    sr_min = min(r for r, c in shape_cells)
    sr_max = max(r for r, c in shape_cells)
    sc_min = min(c for r, c in shape_cells)
    sc_max = max(c for r, c in shape_cells)
    shape_center_r = (sr_min + sr_max) / 2
    shape_center_c = (sc_min + sc_max) / 2
    
    ref_center_r = 2 * center_r - shape_center_r
    ref_center_c = 2 * center_c - shape_center_c
    
    shift_r = ref_center_r - shape_center_r
    shift_c = ref_center_c - shape_center_c
    
    new_cells = set()
    
    for mult in range(1, 50):
        stamp_has_cells = False
        for sr, sc in shape_cells:
            nr = round(sr + mult * shift_r)
            nc = round(sc + mult * shift_c)
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in shape_set and (nr, nc) not in marker_cells:
                    new_cells.add((nr, nc))
                stamp_has_cells = True
        if not stamp_has_cells:
            break
    
    output = [row[:] for row in input_grid]
    for r, c in new_cells:
        output[r][c] = new_color
    
    return output
