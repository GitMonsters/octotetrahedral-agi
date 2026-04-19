import numpy as np

def transform(grid):
    """
    Pattern: Grid is divided by separator lines into tiles.
    One tile contains a pattern (non-background colors).
    That pattern gets copied/tiled across all sections.
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find separator color - forms complete rows or columns
    colors = set(grid.flatten())
    sep_color = None
    for c in colors:
        row_mask = np.all(grid == c, axis=1)
        col_mask = np.all(grid == c, axis=0)
        if np.any(row_mask) or np.any(col_mask):
            sep_color = c
            break
    
    if sep_color is None:
        return grid.tolist()
    
    # Find separator row and column positions
    sep_rows = [i for i in range(h) if np.all(grid[i] == sep_color)]
    sep_cols = [j for j in range(w) if np.all(grid[:, j] == sep_color)]
    
    # Determine tile boundaries
    row_bounds = [-1] + sep_rows + [h]
    col_bounds = [-1] + sep_cols + [w]
    
    # Extract tiles and their positions
    tiles = []
    tile_positions = []
    for ri in range(len(row_bounds) - 1):
        for ci in range(len(col_bounds) - 1):
            r_start = row_bounds[ri] + 1
            r_end = row_bounds[ri + 1]
            c_start = col_bounds[ci] + 1
            c_end = col_bounds[ci + 1]
            if r_end > r_start and c_end > c_start:
                tile = grid[r_start:r_end, c_start:c_end]
                tiles.append(tile)
                tile_positions.append((r_start, r_end, c_start, c_end))
    
    # Find background color (most common in tiles)
    all_values = []
    for tile in tiles:
        all_values.extend(tile.flatten())
    unique, counts = np.unique(all_values, return_counts=True)
    bg_color = unique[np.argmax(counts)]
    
    # Find tile with pattern (has non-background colors)
    pattern_tile = None
    for tile in tiles:
        if np.sum(tile != bg_color) > 0:
            pattern_tile = tile.copy()
            break
    
    if pattern_tile is None:
        return grid.tolist()
    
    # Copy pattern to all tile positions
    result = grid.copy()
    for r_start, r_end, c_start, c_end in tile_positions:
        tile_h = r_end - r_start
        tile_w = c_end - c_start
        if tile_h == pattern_tile.shape[0] and tile_w == pattern_tile.shape[1]:
            result[r_start:r_end, c_start:c_end] = pattern_tile
    
    return result.tolist()
