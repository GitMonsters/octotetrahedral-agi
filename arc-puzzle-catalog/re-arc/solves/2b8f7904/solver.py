def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])

    # Count colors
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            v = input_grid[r][c]
            color_counts[v] = color_counts.get(v, 0) + 1

    bg_color = max(color_counts, key=color_counts.get)
    non_bg = [c for c in color_counts if c != bg_color]

    if len(non_bg) == 0:
        return [row[:] for row in input_grid]

    def find_rect(color):
        min_r = min_c = float('inf')
        max_r = max_c = -1
        count = 0
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    count += 1
        expected = (max_r - min_r + 1) * (max_c - min_c + 1)
        return count == expected, (min_r, max_r, min_c, max_c)

    # Single non-bg color: if it's a solid rectangle, erase it
    if len(non_bg) == 1:
        is_rect, bbox = find_rect(non_bg[0])
        if is_rect:
            output = [row[:] for row in input_grid]
            for r in range(bbox[0], bbox[1] + 1):
                for c in range(bbox[2], bbox[3] + 1):
                    output[r][c] = bg_color
            return output
        return [row[:] for row in input_grid]

    # 2+ non-bg colors: find overlay (the solid rectangle)
    overlay_color = None
    ov_bbox = None
    for color in non_bg:
        is_rect, bbox = find_rect(color)
        if is_rect:
            overlay_color = color
            ov_bbox = bbox
            break

    if overlay_color is None:
        return [row[:] for row in input_grid]

    ov_min_r, ov_max_r, ov_min_c, ov_max_c = ov_bbox

    # Bounding box of all non-background cells
    all_min_r = all_min_c = float('inf')
    all_max_r = all_max_c = -1
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg_color:
                all_min_r = min(all_min_r, r)
                all_max_r = max(all_max_r, r)
                all_min_c = min(all_min_c, c)
                all_max_c = max(all_max_c, c)

    center_r = (all_min_r + all_max_r) / 2.0
    center_c = (all_min_c + all_max_c) / 2.0

    # Check if horizontal mirror maps overlay outside itself
    h_works = True
    for c in range(ov_min_c, ov_max_c + 1):
        mc = int(round(2 * center_c - c))
        if ov_min_c <= mc <= ov_max_c:
            h_works = False
            break

    # Check if vertical mirror maps overlay outside itself
    v_works = True
    for r in range(ov_min_r, ov_max_r + 1):
        mr = int(round(2 * center_r - r))
        if ov_min_r <= mr <= ov_max_r:
            v_works = False
            break

    output = [row[:] for row in input_grid]

    if h_works:
        for r in range(ov_min_r, ov_max_r + 1):
            for c in range(ov_min_c, ov_max_c + 1):
                mc = int(round(2 * center_c - c))
                if 0 <= mc < cols:
                    output[r][c] = input_grid[r][mc]
                else:
                    output[r][c] = bg_color
    elif v_works:
        for r in range(ov_min_r, ov_max_r + 1):
            for c in range(ov_min_c, ov_max_c + 1):
                mr = int(round(2 * center_r - r))
                if 0 <= mr < rows:
                    output[r][c] = input_grid[mr][c]
                else:
                    output[r][c] = bg_color

    return output
