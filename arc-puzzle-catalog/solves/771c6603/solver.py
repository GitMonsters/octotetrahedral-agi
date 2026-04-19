from collections import Counter

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find background color (most frequent)
    bg = Counter(grid[r][c] for r in range(rows) for c in range(cols)).most_common(1)[0][0]
    
    # Find frame rectangle: 4 same-color corners with all-bg interior
    cells_by_color = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != bg:
                cells_by_color.setdefault(v, []).append((r, c))
    
    best_frame = None
    best_area = -1
    corner_color = None
    
    for color, cells in cells_by_color.items():
        by_row = {}
        for r, c in cells:
            by_row.setdefault(r, set()).add(c)
        row_list = sorted(by_row.keys())
        for i in range(len(row_list)):
            for j in range(i + 1, len(row_list)):
                r1, r2 = row_list[i], row_list[j]
                common = sorted(by_row[r1] & by_row[r2])
                if len(common) < 2:
                    continue
                c1, c2 = common[0], common[-1]
                if r2 - r1 < 2 or c2 - c1 < 2:
                    continue
                # Check interior is all bg
                all_bg_inside = True
                for ir in range(r1 + 1, r2):
                    for ic in range(c1 + 1, c2):
                        if grid[ir][ic] != bg:
                            all_bg_inside = False
                            break
                    if not all_bg_inside:
                        break
                if all_bg_inside:
                    area = (r2 - r1) * (c2 - c1)
                    if area > best_area:
                        best_area = area
                        best_frame = (r1, c1, r2, c2)
                        corner_color = color
    
    r1, c1, r2, c2 = best_frame
    out_h = r2 - r1 + 1
    out_w = c2 - c1 + 1
    
    # Build output with frame border
    output = [[bg] * out_w for _ in range(out_h)]
    for c in range(c1, c2 + 1):
        output[0][c - c1] = grid[r1][c]
        output[out_h - 1][c - c1] = grid[r2][c]
    for r in range(r1, r2 + 1):
        output[r - r1][0] = grid[r][c1]
        output[r - r1][out_w - 1] = grid[r][c2]
    
    # Determine decorated borders (non-bg, non-corner-color fill)
    def is_decorated(cells):
        return any(v != bg and v != corner_color for v in cells)
    
    top_dec = is_decorated([grid[r1][c] for c in range(c1 + 1, c2)])
    bottom_dec = is_decorated([grid[r2][c] for c in range(c1 + 1, c2)])
    left_dec = is_decorated([grid[r][c1] for r in range(r1 + 1, r2)])
    right_dec = is_decorated([grid[r][c2] for r in range(r1 + 1, r2)])
    
    # Find content cells (non-bg, NOT on frame border)
    frame_border = set()
    for c in range(c1, c2 + 1):
        frame_border.add((r1, c))
        frame_border.add((r2, c))
    for r in range(r1 + 1, r2):
        frame_border.add((r, c1))
        frame_border.add((r, c2))
    
    content_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in frame_border:
                content_cells.append((r, c, grid[r][c]))
    
    if not content_cells:
        return output
    
    ct_min_r = min(r for r, c, v in content_cells)
    ct_max_r = max(r for r, c, v in content_cells)
    ct_min_c = min(c for r, c, v in content_cells)
    ct_max_c = max(c for r, c, v in content_cells)
    ct_h = ct_max_r - ct_min_r + 1
    ct_w = ct_max_c - ct_min_c + 1
    
    content = [[bg] * ct_w for _ in range(ct_h)]
    for r, c, v in content_cells:
        content[r - ct_min_r][c - ct_min_c] = v
    
    # Flip based on decorated borders
    flip_v = (bottom_dec != top_dec)
    flip_h = (left_dec != right_dec)
    
    if flip_v:
        content = content[::-1]
    if flip_h:
        content = [row[::-1] for row in content]
    
    # Placement in interior
    int_h = out_h - 2
    int_w = out_w - 2
    
    # Vertical offset
    if bottom_dec and not top_dec:
        v_off = int_h - ct_h
    elif top_dec and not bottom_dec:
        v_off = 0
    else:
        ct_center_r = (ct_min_r + ct_max_r) / 2
        frame_center_r = (r1 + r2) / 2
        v_off = 0 if ct_center_r < frame_center_r else int_h - ct_h
    
    # Horizontal offset
    if left_dec and not right_dec:
        h_off = 0
    elif right_dec and not left_dec:
        h_off = int_w - ct_w
    else:
        ct_center_c = (ct_min_c + ct_max_c) / 2
        frame_center_c = (c1 + c2) / 2
        h_off = 0 if ct_center_c < frame_center_c else int_w - ct_w
    
    # Place content
    for r in range(ct_h):
        for c in range(ct_w):
            out_r = 1 + v_off + r
            out_c = 1 + h_off + c
            if 0 < out_r < out_h - 1 and 0 < out_c < out_w - 1:
                output[out_r][out_c] = content[r][c]
    
    return output
