def transform(input_grid):
    import numpy as np
    grid = np.array(input_grid)
    rows, cols = grid.shape

    # Background = most common value
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])

    # Find separator: a full row or column of a single non-bg value
    sep_row = sep_col = None
    for r in range(rows):
        v = int(grid[r, 0])
        if v != bg and np.all(grid[r] == v):
            sep_row = r
            break
    if sep_row is None:
        for c in range(cols):
            v = int(grid[0, c])
            if v != bg and np.all(grid[:, c] == v):
                sep_col = c
                break

    # Split into template side (smaller) and marker side (larger)
    if sep_row is not None:
        above = grid[:sep_row]
        below = grid[sep_row + 1:]
        if above.shape[0] <= below.shape[0]:
            template_grid, marker_grid = above, below
            m_r_off, m_c_off = sep_row + 1, 0
        else:
            template_grid, marker_grid = below, above
            m_r_off, m_c_off = 0, 0
    else:
        left = grid[:, :sep_col]
        right = grid[:, sep_col + 1:]
        if left.shape[1] <= right.shape[1]:
            template_grid, marker_grid = left, right
            m_r_off, m_c_off = 0, sep_col + 1
        else:
            template_grid, marker_grid = right, left
            m_r_off, m_c_off = 0, 0

    # Extract tile from bounding box of non-bg pixels on template side
    non_bg = np.argwhere(template_grid != bg)
    r_min, c_min = non_bg.min(axis=0)
    r_max, c_max = non_bg.max(axis=0)
    tile = template_grid[r_min:r_max + 1, c_min:c_max + 1]
    tile_h, tile_w = tile.shape
    cr, cc = tile_h // 2, tile_w // 2  # center of tile

    # Find marker positions on marker side
    markers = np.argwhere(marker_grid != bg)

    # Stamp tile centered at each marker position
    output = grid.copy()
    for mr, mc in markers:
        ar, ac = m_r_off + mr, m_c_off + mc
        sr, sc = ar - cr, ac - cc
        for tr in range(tile_h):
            for tc in range(tile_w):
                out_r, out_c = sr + tr, sc + tc
                if 0 <= out_r < rows and 0 <= out_c < cols:
                    output[out_r, out_c] = tile[tr, tc]

    return output.tolist()
