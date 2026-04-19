def _get_background_color(grid):
    """Find the most common color (background)."""
    color_counts = {}
    for row in grid:
        for val in row:
            color_counts[val] = color_counts.get(val, 0) + 1
    return max(color_counts, key=color_counts.get)


def _get_foreground_pixels(grid, background):
    """Return list of (r, c) coordinates for non-background pixels."""
    pixels = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != background:
                pixels.append((r, c))
    return pixels


def _block_has_foreground(grid, background, min_r, min_c, block_h, block_w, block_row, block_col):
    """Check if a block position in the 3x3 grid has foreground."""
    rows, cols = len(grid), len(grid[0])
    start_r = min_r + block_row * block_h
    start_c = min_c + block_col * block_w
    for r in range(start_r, start_r + block_h):
        for c in range(start_c, start_c + block_w):
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] != background:
                return True
    return False


def transform(grid):
    """
    Find the foreground pattern, interpret its bounding box as a 3x3 grid
    of 3x3 blocks, count filled edge positions (top-middle, middle-left,
    middle-right, bottom-middle), and return [[5 + (4 - edge_count)//2]].
    Returns [[5]] if the grid is uniform.
    """
    background = _get_background_color(grid)
    fg_pixels = _get_foreground_pixels(grid, background)
    
    if not fg_pixels:
        return [[5]]
    
    min_r = min(p[0] for p in fg_pixels)
    max_r = max(p[0] for p in fg_pixels)
    min_c = min(p[1] for p in fg_pixels)
    max_c = max(p[1] for p in fg_pixels)
    
    block_h = (max_r - min_r + 1) // 3
    block_w = (max_c - min_c + 1) // 3
    
    if block_h == 0 or block_w == 0:
        return [[5]]
    
    # Edge positions in a 3x3 grid: top-middle, middle-left, middle-right, bottom-middle
    edge_positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
    edge_count = sum(
        1 for br, bc in edge_positions
        if _block_has_foreground(grid, background, min_r, min_c, block_h, block_w, br, bc)
    )
    
    return [[5 + (4 - edge_count) // 2]]
