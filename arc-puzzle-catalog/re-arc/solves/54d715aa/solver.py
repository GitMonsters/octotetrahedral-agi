def transform(input_grid):
    import copy
    from collections import Counter

    rows = len(input_grid)
    cols = len(input_grid[0])
    output = copy.deepcopy(input_grid)

    # Find background color (most common)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find non-background pixels
    pixels = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                pixels.append((r, c, input_grid[r][c]))

    # Each pixel generates a 3-row checkerboard band
    for pr, pc, color in pixels:
        parity = (pr + pc) % 2
        for r in range(max(0, pr - 1), min(rows, pr + 2)):
            for c in range(cols):
                if (r + c) % 2 == parity:
                    output[r][c] = color

    return output
