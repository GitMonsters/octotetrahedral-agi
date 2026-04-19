def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    # Find background color (most common)
    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            color_count[input_grid[r][c]] += 1
    bg = color_count.most_common(1)[0][0]

    # Find all non-background colors and their pixels
    color_pixels = {}
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                color = input_grid[r][c]
                if color not in color_pixels:
                    color_pixels[color] = []
                color_pixels[color].append((r, c))

    if len(color_pixels) < 2:
        return [row[:] for row in input_grid]

    # Identify shape (more pixels) and arrow (fewer pixels)
    sorted_colors = sorted(color_pixels.items(), key=lambda x: len(x[1]), reverse=True)
    shape_color = sorted_colors[0][0]
    shape_pixels_list = sorted_colors[0][1]
    arrow_color = sorted_colors[1][0]
    arrow_pixels_list = sorted_colors[1][1]

    shape_pixels_set = set(shape_pixels_list)

    # Shape bounding box
    min_r = min(r for r, c in shape_pixels_list)
    max_r = max(r for r, c in shape_pixels_list)
    min_c = min(c for r, c in shape_pixels_list)
    max_c = max(c for r, c in shape_pixels_list)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Extract boolean pattern (True = shape foreground pixel)
    pattern = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append((r, c) in shape_pixels_set)
        pattern.append(row)

    # Direction: from shape center to arrow centroid
    shape_cr = (min_r + max_r) / 2.0
    shape_cc = (min_c + max_c) / 2.0
    ar = sum(r for r, c in arrow_pixels_list) / len(arrow_pixels_list)
    ac = sum(c for r, c in arrow_pixels_list) / len(arrow_pixels_list)

    dr = 1 if ar > shape_cr else -1
    dc = 1 if ac > shape_cc else -1

    # Corner of shape in the arrow direction
    corner_r_idx = h - 1 if dr > 0 else 0
    corner_c_idx = w - 1 if dc > 0 else 0

    # If corner pixel is bg, copies overlap at bg corner (shift = size-1)
    # If corner pixel is foreground, copies have 1-cell gap (shift = size)
    if pattern[corner_r_idx][corner_c_idx]:
        shift_r = h * dr
        shift_c = w * dc
    else:
        shift_r = (h - 1) * dr
        shift_c = (w - 1) * dc

    if shift_r == 0 and shift_c == 0:
        return [row[:] for row in input_grid]

    output = [row[:] for row in input_grid]

    # Generate copies starting from first shifted position
    curr_r = min_r + shift_r
    curr_c = min_c + shift_c

    for _ in range(rows + cols):  # safety limit
        c_max_r = curr_r + h - 1
        c_max_c = curr_c + w - 1
        if c_max_r < 0 or curr_r >= rows or c_max_c < 0 or curr_c >= cols:
            break

        for pr in range(h):
            for pc in range(w):
                if pattern[pr][pc]:
                    gr = curr_r + pr
                    gc = curr_c + pc
                    if 0 <= gr < rows and 0 <= gc < cols:
                        output[gr][gc] = arrow_color

        curr_r += shift_r
        curr_c += shift_c

    return output
