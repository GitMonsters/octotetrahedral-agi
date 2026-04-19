def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    Rule: The input has a background, a small multi-colored pattern (key), and
    a large single-color shape. The shape is an integer multiple of the pattern
    size. Downscale the shape to the pattern size; where the shape is present,
    keep the pattern color; where absent, fill with the background color.
    """
    import numpy as np
    from collections import Counter

    g = np.array(grid)
    colors = Counter(g.flatten())
    bg = colors.most_common(1)[0][0]
    shape_color = colors.most_common(2)[1][0]

    # Pattern = bounding box of "rare" colors (not bg, not shape_color)
    rare_mask = (g != bg) & (g != shape_color)
    rare_pos = np.argwhere(rare_mask)
    pr_min, pc_min = rare_pos.min(axis=0)
    pr_max, pc_max = rare_pos.max(axis=0)
    pattern = g[pr_min:pr_max + 1, pc_min:pc_max + 1]
    pat_h, pat_w = pattern.shape

    # Shape = all shape_color pixels outside the pattern region
    shape_mask = (g == shape_color).copy()
    shape_mask[pr_min:pr_max + 1, pc_min:pc_max + 1] = False
    shape_pos = np.argwhere(shape_mask)
    sr_min, sc_min = shape_pos.min(axis=0)
    sr_max, sc_max = shape_pos.max(axis=0)

    row_factor = (sr_max - sr_min + 1) // pat_h
    col_factor = (sc_max - sc_min + 1) // pat_w

    result = np.full((pat_h, pat_w), bg, dtype=int)
    for r in range(pat_h):
        for c in range(pat_w):
            br = sr_min + r * row_factor
            bc = sc_min + c * col_factor
            block = g[br:br + row_factor, bc:bc + col_factor]
            if np.all(block == shape_color):
                result[r, c] = pattern[r, c]

    return result.tolist()
