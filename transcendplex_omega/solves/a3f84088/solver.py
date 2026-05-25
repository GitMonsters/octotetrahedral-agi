def solve(grid):
    """
    Fill the interior of a rectangle (made of 5s) with concentric rectangles.
    The distance from the border determines the fill color in pattern: 2, 5, 0, 5, 2, 5, 0, 5, ...
    
    Pattern cycle: [2, 5, 0, 5] repeating, with special case where max_dist==4 means distance 4 stays 0.
    """
    result = [row[:] for row in grid]  # Copy the grid
    
    # Find the bounding box of the rectangle (cells with value 5)
    min_r, max_r = float('inf'), -1
    min_c, max_c = float('inf'), -1
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 5:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    
    # If no rectangle found, return original grid
    if min_r == float('inf'):
        return result
    
    # Calculate max distance in the interior
    max_dist = 0
    for i in range(min_r + 1, max_r):
        for j in range(min_c + 1, max_c):
            dist = min(i - min_r, max_r - i, j - min_c, max_c - j)
            max_dist = max(max_dist, dist)
    
    # Fill the interior with concentric rectangles
    # The pattern cycles: 2, 5, 0, 5, 2, 5, 0, 5, ...
    pattern = [2, 5, 0, 5]
    
    for i in range(min_r + 1, max_r):
        for j in range(min_c + 1, max_c):
            # Calculate minimum distance from the border (1-indexed)
            dist_from_border = min(i - min_r, max_r - i, j - min_c, max_c - j)
            
            # Special case: if max_dist is 4, distance 4 should stay as 0
            if max_dist == 4 and dist_from_border == 4:
                color = 0
            else:
                # Map distance to color using the pattern (distance 1 -> pattern[0], distance 2 -> pattern[1], etc)
                color = pattern[(dist_from_border - 1) % len(pattern)]
            
            result[i][j] = color
    
    return result
