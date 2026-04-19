def transform(grid):
    """
    ARC puzzle 7e40550c solver.
    
    Transformation rule (empirically determined):
    - Identify the two colors in the grid: majority (more frequent) and minority
    - Cells of majority color become color 5 if they are at Manhattan distance
      1-4 from any cell of the minority color
    - All other cells remain unchanged
    
    Accuracy: ~80% on training examples (261/1304 errors)
    This rule captures the core pattern but there are edge cases not fully understood
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    arr = np.array(grid)
    colors = np.unique(arr)
    
    if len(colors) != 2:
        return arr.tolist()
    
    color1, color2 = colors[0], colors[1]
    count1 = np.sum(arr == color1)
    count2 = np.sum(arr == color2)
    
    # Identify majority and minority colors
    if count1 > count2:
        majority, minority = color1, color2
    else:
        majority, minority = color2, color1
    
    # Calculate Manhattan distance from each cell to the nearest minority cell
    dist_to_minor = distance_transform_edt(arr != minority)
    
    # Mark majority cells that are distance 1-4 from minority as becoming color 5
    becomes_5 = (arr == majority) & (dist_to_minor >= 1) & (dist_to_minor <= 4)
    
    result = arr.copy()
    result[becomes_5] = 5
    
    return result.tolist()
