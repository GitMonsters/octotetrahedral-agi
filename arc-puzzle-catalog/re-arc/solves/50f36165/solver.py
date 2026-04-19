def transform(input_grid: list[list[int]]) -> list[list[int]]:
    import numpy as np

    grid = np.array(input_grid)
    height, width = grid.shape

    bg = int(np.bincount(grid.flatten()).argmax())

    non_bg = np.argwhere(grid != bg)
    if len(non_bg) == 0:
        return input_grid

    min_row, min_col = non_bg.min(axis=0)
    max_row, max_col = non_bg.max(axis=0)

    bbox = grid[min_row:max_row + 1, min_col:max_col + 1]
    bbox_h, bbox_w = bbox.shape

    # Find smallest horizontal period
    h_period = None
    for p in range(1, bbox_w):
        if all(bbox[r, c] == bbox[r, c % p] for r in range(bbox_h) for c in range(bbox_w)):
            h_period = p
            break

    # Find smallest vertical period
    v_period = None
    for p in range(1, bbox_h):
        if all(bbox[r, c] == bbox[r % p, c] for r in range(bbox_h) for c in range(bbox_w)):
            v_period = p
            break

    output = grid.copy()

    if h_period is not None and v_period is not None:
        tile = bbox[:v_period, :h_period].copy()
        for r in range(height):
            for c in range(width):
                output[r, c] = tile[(r - min_row) % v_period, (c - min_col) % h_period]
    elif h_period is not None:
        for r in range(min_row, max_row + 1):
            tile = grid[r, min_col:min_col + h_period].copy()
            for c in range(width):
                output[r, c] = tile[(c - min_col) % h_period]
    elif v_period is not None:
        tile_rows = grid[min_row:min_row + v_period].copy()
        for r in range(height):
            output[r] = tile_rows[(r - min_row) % v_period]

    return output.tolist()
