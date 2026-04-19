from collections import Counter


def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1

    bg = color_counts.most_common(1)[0][0]
    non_bg = [c for c in color_counts if c != bg]

    # No overlay possible with fewer than 2 non-bg colors
    if len(non_bg) <= 1:
        return [row[:] for row in grid]

    # Overlay color fills a perfect rectangle; shape color does not
    overlay_color = None
    shape_color = None

    for color in non_bg:
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        expected = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(cells) == expected:
            overlay_color = color
        else:
            shape_color = color

    if overlay_color is None:
        return [row[:] for row in grid]
    if shape_color is None:
        for c in non_bg:
            if c != overlay_color:
                shape_color = c
                break

    # Bounding box of shape color only (not overlay)
    shape_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color]
    s_min_r = min(r for r, c in shape_cells)
    s_max_r = max(r for r, c in shape_cells)
    s_min_c = min(c for r, c in shape_cells)
    s_max_c = max(c for r, c in shape_cells)

    center_r = (s_min_r + s_max_r) / 2.0
    center_c = (s_min_c + s_max_c) / 2.0

    def mirror_v(r, c):
        return (r, int(round(2 * center_c - c)))

    def mirror_h(r, c):
        return (int(round(2 * center_r - r)), c)

    # Check which axis makes shape cells symmetric (mirror lands on shape or overlay)
    def check_sym(mirror_fn):
        for r, c in shape_cells:
            mr, mc = mirror_fn(r, c)
            if not (0 <= mr < rows and 0 <= mc < cols):
                return False
            if grid[mr][mc] != shape_color and grid[mr][mc] != overlay_color:
                return False
        return True

    v_sym = check_sym(mirror_v)
    h_sym = check_sym(mirror_h)

    if v_sym and not h_sym:
        mirror = mirror_v
    elif h_sym and not v_sym:
        mirror = mirror_h
    elif v_sym and h_sym:
        ov_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == overlay_color]
        ov_cr = sum(r for r, c in ov_cells) / len(ov_cells)
        ov_cc = sum(c for r, c in ov_cells) / len(ov_cells)
        mirror = mirror_v if abs(ov_cc - center_c) > abs(ov_cr - center_r) else mirror_h
    else:
        mirror = mirror_v

    # Replace each overlay cell with its mirror value
    result = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == overlay_color:
                mr, mc = mirror(r, c)
                if 0 <= mr < rows and 0 <= mc < cols:
                    result[r][c] = grid[mr][mc]
                else:
                    result[r][c] = bg

    return result
