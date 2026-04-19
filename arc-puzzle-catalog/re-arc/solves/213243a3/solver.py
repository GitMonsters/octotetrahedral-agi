def transform(input_grid):
    from collections import Counter

    # Find background color (most frequent)
    flat = [c for row in input_grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Count green (3) cells — this determines how many blue (1) cells in output
    n = flat.count(3)

    # Create 3×3 background grid
    output = [[bg] * 3 for _ in range(3)]

    # Fill order: top-right going left across row 0, then center of row 1, etc.
    fill_order = [
        (0, 2), (0, 1), (0, 0),
        (1, 1),
        (1, 2), (1, 0),
        (2, 1), (2, 2), (2, 0),
    ]

    for i in range(min(n, 9)):
        r, c = fill_order[i]
        output[r][c] = 1

    return output
