def transform(input_grid):
    import numpy as np
    grid = np.array(input_grid)

    # Find background color (most common)
    unique, counts = np.unique(grid, return_counts=True)
    bg = unique[np.argmax(counts)]

    # Find all non-background positions
    non_bg = np.argwhere(grid != bg)
    min_r, min_c = non_bg.min(axis=0)
    max_r, max_c = non_bg.max(axis=0)

    # Check for a rectangular frame: 4 corner markers with background edges between them
    corners_nonbg = all(
        grid[r, c] != bg for r, c in
        [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]
    )

    has_frame = False
    if corners_nonbg and max_r - min_r >= 2 and max_c - min_c >= 2:
        top = grid[min_r, min_c + 1:max_c]
        bot = grid[max_r, min_c + 1:max_c]
        left = grid[min_r + 1:max_r, min_c]
        right = grid[min_r + 1:max_r, max_c]
        if (np.all(top == bg) and np.all(bot == bg) and
                np.all(left == bg) and np.all(right == bg)):
            has_frame = True

    if has_frame:
        corner_color = int(grid[min_r, min_c])
        interior = grid[min_r + 1:max_r, min_c + 1:max_c].copy()
        interior[interior != bg] = corner_color
        return interior.tolist()
    else:
        return grid[min_r:max_r + 1, min_c:max_c + 1].tolist()
