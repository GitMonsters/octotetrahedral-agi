def transform(input_grid):
    nrows = len(input_grid)
    ncols = len(input_grid[0])
    output = [row[:] for row in input_grid]
    center = ncols - nrows - 1  # 12 for 5x18

    for r in range(nrows):
        for c in range(ncols):
            # Left triangle: top-left corner, shrinking downward
            if r + c <= nrows - 2:
                output[r][c] = 2
            # Right triangle: expanding downward from apex at (0, center)
            elif r >= 1 and abs(c - center) <= r - 1:
                output[r][c] = 2

    return output
