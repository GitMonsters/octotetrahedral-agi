from collections import Counter

def transform(grid):
    """
    ARC puzzle 2c3c6858: For each row, if the marker pixel (non-background) 
    is in the left third of the grid, fill the row with 4 (yellow).
    Otherwise, fill with 9 (maroon).
    """
    height = len(grid)
    width = len(grid[0])
    threshold = width // 3
    
    # Find background color (most common)
    all_vals = [v for row in grid for v in row]
    bg_color = Counter(all_vals).most_common(1)[0][0]
    
    result = []
    for row in grid:
        # Find marker position (first non-background pixel)
        marker_col = None
        for col, val in enumerate(row):
            if val != bg_color:
                marker_col = col
                break
        
        # If marker in left third, row becomes yellow (4), else maroon (9)
        if marker_col is not None and marker_col < threshold:
            result.append([4] * width)
        else:
            result.append([9] * width)
    
    return result
