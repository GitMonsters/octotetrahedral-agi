def transform(input_grid):
    from collections import Counter

    R = len(input_grid)
    C = len(input_grid[0])

    # Find cells that differ under 180° rotation
    diff_cells = []
    for r in range(R):
        for c in range(C):
            rr, cc = R - 1 - r, C - 1 - c
            if input_grid[r][c] != input_grid[rr][cc]:
                diff_cells.append((r, c))

    # The paint color is the most common value among asymmetric cells
    color_counts = Counter(input_grid[r][c] for r, c in diff_cells)
    paint_color = color_counts.most_common(1)[0][0]

    # Painted cells: asymmetric cells whose value is the paint color
    painted_cells = [(r, c) for r, c in diff_cells if input_grid[r][c] == paint_color]

    # Bounding box of painted region
    min_r = min(r for r, c in painted_cells)
    max_r = max(r for r, c in painted_cells)
    min_c = min(c for r, c in painted_cells)
    max_c = max(c for r, c in painted_cells)

    # Reconstruct original content using the 180°-rotated positions
    output = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            rr, cc = R - 1 - r, C - 1 - c
            row.append(input_grid[rr][cc])
        output.append(row)

    return output
