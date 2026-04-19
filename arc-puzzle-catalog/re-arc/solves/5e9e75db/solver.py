import numpy as np

def transform(grid):
    """
    The grid has point symmetry (180° rotational symmetry).
    The yellow (4) region marks a "damaged" area.
    To find what should be there, we look at the point-symmetric opposite region
    and rotate it 180 degrees.
    """
    grid = np.array(grid)
    H, W = grid.shape
    
    # Find yellow (4) region
    yellow_mask = grid == 4
    rows = np.where(yellow_mask.any(axis=1))[0]
    cols = np.where(yellow_mask.any(axis=0))[0]
    r1, r2 = rows[0], rows[-1]
    c1, c2 = cols[0], cols[-1]
    
    # Find the point-symmetric opposite region
    opp_r1 = H - 1 - r2
    opp_r2 = H - 1 - r1
    opp_c1 = W - 1 - c2
    opp_c2 = W - 1 - c1
    
    # Extract and rotate 180 degrees
    opposite_region = grid[opp_r1:opp_r2+1, opp_c1:opp_c2+1]
    result = np.rot90(opposite_region, 2)
    
    return result.tolist()
