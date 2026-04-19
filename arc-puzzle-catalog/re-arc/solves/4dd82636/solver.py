def transform(grid):
    """
    Puzzle 4dd82636: Find repeating tile pattern and tile it across the full grid.
    The input has a region filled with background color - extract the pattern tile
    from the non-background region and repeat it to fill the output.
    """
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the background color (most common color in the grid that forms a contiguous region)
    # Check corners and edges to identify background
    unique, counts = np.unique(grid, return_counts=True)
    
    # Find background color - it's the one that fills entire rows or columns
    bg_color = None
    for color in unique:
        # Check if any entire row is this color
        for r in range(h):
            if np.all(grid[r, :] == color):
                bg_color = color
                break
        if bg_color is not None:
            break
        # Check if any entire column is this color
        for c in range(w):
            if np.all(grid[:, c] == color):
                bg_color = color
                break
        if bg_color is not None:
            break
    
    if bg_color is None:
        # Find color that dominates a corner region
        for color in unique:
            mask = (grid == color)
            # Check top-left corner
            if mask[0, 0] and mask[0, 1] and mask[1, 0]:
                bg_color = color
                break
            # Check bottom-right corner
            if mask[-1, -1] and mask[-1, -2] and mask[-2, -1]:
                bg_color = color
                break
    
    # Find the bounding box of non-background region
    mask = grid != bg_color
    rows_with_pattern = np.any(mask, axis=1)
    cols_with_pattern = np.any(mask, axis=0)
    
    if not np.any(rows_with_pattern):
        return grid.tolist()
    
    r_min = np.argmax(rows_with_pattern)
    r_max = h - 1 - np.argmax(rows_with_pattern[::-1])
    c_min = np.argmax(cols_with_pattern)
    c_max = w - 1 - np.argmax(cols_with_pattern[::-1])
    
    # Extract pattern region
    pattern_region = grid[r_min:r_max+1, c_min:c_max+1]
    ph, pw = pattern_region.shape
    
    # Find the repeating tile period
    def find_period(arr, axis):
        size = arr.shape[axis]
        for period in range(1, size + 1):
            if size % period == 0:
                continue  # Check all periods, not just divisors
            valid = True
            for i in range(size):
                if axis == 0:
                    if not np.array_equal(arr[i, :], arr[i % period, :]):
                        valid = False
                        break
                else:
                    if not np.array_equal(arr[:, i], arr[:, i % period]):
                        valid = False
                        break
            if valid:
                return period
        # Try divisors
        for period in range(1, size + 1):
            valid = True
            for i in range(size):
                if axis == 0:
                    if not np.array_equal(arr[i, :], arr[i % period, :]):
                        valid = False
                        break
                else:
                    if not np.array_equal(arr[:, i], arr[:, i % period]):
                        valid = False
                        break
            if valid:
                return period
        return size
    
    tile_h = find_period(pattern_region, 0)
    tile_w = find_period(pattern_region, 1)
    
    # Extract the base tile
    tile = pattern_region[:tile_h, :tile_w]
    
    # Create output by tiling
    output = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            output[r, c] = tile[r % tile_h, c % tile_w]
    
    return output.tolist()
