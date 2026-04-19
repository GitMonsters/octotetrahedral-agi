def transform(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Check if grid is uniform (all one color)
    bg = grid[0][0]
    all_same = all(grid[r][c] == bg for r in range(rows) for c in range(cols))

    if all_same:
        # Add a minimal vertical line at column (rows-2), from row (rows-2) to bottom
        output = [row[:] for row in grid]
        col_pos = rows - 2
        new_color = (bg + 1) % 10
        for r in range(col_pos, rows):
            output[r][col_pos] = new_color
        return output
    else:
        return [row[:] for row in grid]
