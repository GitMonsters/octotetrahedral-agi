def transform(input_grid):
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    color_counts = Counter(c for c in flat if c != bg)

    center_r = center_c = center_color = None

    # Find center: color with exactly 4 cells forming a 2×2
    for color, count in color_counts.items():
        if count == 4:
            cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid[r][c] == color]
            mr = min(r for r, c in cells)
            mc = min(c for r, c in cells)
            if set(cells) == {(mr, mc), (mr, mc+1), (mr+1, mc), (mr+1, mc+1)}:
                center_color = color
                center_r = mr
                center_c = mc
                break

    # Fallback: find the only 2×2 filled block of non-bg color
    if center_color is None:
        for r in range(rows - 1):
            for c in range(cols - 1):
                v = input_grid[r][c]
                if v != bg and v == input_grid[r][c+1] == input_grid[r+1][c] == input_grid[r+1][c+1]:
                    center_color = v
                    center_r = r
                    center_c = c
                    break
            if center_color is not None:
                break

    # Integer doubled center coords (avoids floating point)
    cr2 = 2 * center_r + 1
    cc2 = 2 * center_c + 1

    center_cells = {(center_r, center_c), (center_r, center_c+1),
                    (center_r+1, center_c), (center_r+1, center_c+1)}

    output = copy.deepcopy(input_grid)

    # Reflect all pattern cells (non-bg, non-center) in 4 quadrants
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg and (r, c) not in center_cells:
                color = input_grid[r][c]
                hc = cc2 - c   # horizontal reflection
                vr = cr2 - r   # vertical reflection

                if 0 <= hc < cols:
                    output[r][hc] = color
                if 0 <= vr < rows:
                    output[vr][c] = color
                if 0 <= vr < rows and 0 <= hc < cols:
                    output[vr][hc] = color

    return output
