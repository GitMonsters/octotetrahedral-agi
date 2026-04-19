def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    non_bg = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                non_bg.append((r, c, input_grid[r][c]))

    output = copy.deepcopy(input_grid)

    if len(non_bg) == 0:
        return output

    if len(non_bg) == 1:
        r, c, color = non_bg[0]
        if r == 0 or r == rows - 1:
            for row in range(rows):
                output[row][c] = color
        elif c == 0 or c == cols - 1:
            for col in range(cols):
                output[r][col] = color
        return output

    (r1, c1, color1), (r2, c2, color2) = non_bg[0], non_bg[1]

    on_horiz1 = (r1 == 0 or r1 == rows - 1)
    on_horiz2 = (r2 == 0 or r2 == rows - 1)

    if on_horiz1 or on_horiz2:
        # Pixels on horizontal edges define vertical column lines
        p1, p2 = min(c1, c2), max(c1, c2)
        color_p1 = color1 if c1 <= c2 else color2
        color_p2 = color2 if c1 <= c2 else color1
        d = p2 - p1
        grid_size = cols

        lines = {p1: color_p1, p2: color_p2}

        before = p1 - d
        if before >= 0 and before <= 1:
            lines[before] = color_p2

        after = p2 + d
        if after < grid_size and (grid_size - 1 - after) <= 1:
            lines[after] = color_p1

        for pos, color in lines.items():
            for row in range(rows):
                output[row][pos] = color
    else:
        # Pixels on vertical edges define horizontal row lines
        p1, p2 = min(r1, r2), max(r1, r2)
        color_p1 = color1 if r1 <= r2 else color2
        color_p2 = color2 if r1 <= r2 else color1
        d = p2 - p1
        grid_size = rows

        lines = {p1: color_p1, p2: color_p2}

        before = p1 - d
        if before >= 0 and before <= 1:
            lines[before] = color_p2

        after = p2 + d
        if after < grid_size and (grid_size - 1 - after) <= 1:
            lines[after] = color_p1

        for pos, color in lines.items():
            for col in range(cols):
                output[pos][col] = color

    return output
