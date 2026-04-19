def transform(input_grid: list[list[int]]) -> list[list[int]]:
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    third = cols // 3

    # Find background color
    flat = [c for r in input_grid for c in r]
    bg = Counter(flat).most_common(1)[0][0]

    # Color mapping: left third → 5, middle third → 4, right third → 3
    color_map = {0: 5, 1: 4, 2: 3}

    output = []
    for r in range(rows):
        # Find non-background pixel column
        marker_col = next(c for c in range(cols) if input_grid[r][c] != bg)
        region = marker_col // third
        fill_color = color_map[region]
        output.append([fill_color] * cols)

    return output
