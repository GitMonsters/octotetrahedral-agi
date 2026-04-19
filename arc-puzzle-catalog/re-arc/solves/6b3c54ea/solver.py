def transform(input_grid):
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])

    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find vertical lines: column segments of non-bg color, length >= 2,
    # touching top (start==0) or bottom (end==rows-1) edge
    vertical_lines = {}
    for c in range(cols):
        r = 0
        while r < rows:
            if input_grid[r][c] != bg:
                color = input_grid[r][c]
                start = r
                while r < rows and input_grid[r][c] == color:
                    r += 1
                end = r - 1
                if (end - start + 1) >= 2 and (start == 0 or end == rows - 1):
                    vertical_lines[c] = color
            else:
                r += 1

    # Find horizontal lines: row segments of non-bg color, length >= 2,
    # touching left (start==0) or right (end==cols-1) edge
    horizontal_lines = {}
    for r in range(rows):
        c = 0
        while c < cols:
            if input_grid[r][c] != bg:
                color = input_grid[r][c]
                start = c
                while c < cols and input_grid[r][c] == color:
                    c += 1
                end = c - 1
                if (end - start + 1) >= 2 and (start == 0 or end == cols - 1):
                    horizontal_lines[r] = color
            else:
                c += 1

    # Build output
    output = [[bg] * cols for _ in range(rows)]

    for c, color in vertical_lines.items():
        for r in range(rows):
            output[r][c] = color

    for r, color in horizontal_lines.items():
        for c_idx in range(cols):
            output[r][c_idx] = color

    for r in horizontal_lines:
        for c in vertical_lines:
            output[r][c] = 7

    return output
