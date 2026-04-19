import numpy as np

def transform(grid):
    """
    The grid has point symmetry (180-degree rotational symmetry).
    Some region is corrupted with 7s that break this symmetry.
    Find the 7s that don't have matching 7s at their symmetric position,
    then extract the correct values from the symmetric region (rotated 180°).
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find positions with 7 that break symmetry (no 7 at symmetric point)
    sevens = np.argwhere(grid == 7)
    extra_7s = []
    for r, c in sevens:
        sym_r, sym_c = h - 1 - r, w - 1 - c
        if grid[sym_r, sym_c] != 7:
            extra_7s.append((r, c))
    
    if not extra_7s:
        return grid.tolist()
    
    extra_7s = np.array(extra_7s)
    min_r, min_c = extra_7s.min(axis=0)
    max_r, max_c = extra_7s.max(axis=0)
    
    # Get the symmetric (uncorrupted) region
    sym_min_r, sym_min_c = h - 1 - max_r, w - 1 - max_c
    sym_max_r, sym_max_c = h - 1 - min_r, w - 1 - min_c
    
    extracted = grid[sym_min_r:sym_max_r+1, sym_min_c:sym_max_c+1]
    
    # Rotate 180 degrees to get the original values for the corrupted region
    result = np.rot90(extracted, 2)
    
    return result.tolist()
