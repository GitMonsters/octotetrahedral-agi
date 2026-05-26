# TASK: 7ba870ac
# R1 SCALE:     output is always 3x3
# R2 STRUCTURE: scattered pixels
# R3 SYMMETRY:  irrelevant
# R4 COLOR:     bg = most frequent, markers = other colors
# R5 OBJECTS:   Individual pixels scattered across grid
# RULE: Find bounding box, divide into 3x3, place each pixel in its grid cell
# VERIFIED:     train0 ✓   train1 ✓   train2 ✓

from collections import Counter

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    # Find background
    cc = Counter()
    for row in grid:
        cc.update(row)
    bg = cc.most_common(1)[0][0]
    
    # Find all non-bg pixels
    pixels = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                pixels.append((r, c, grid[r][c]))
    
    if not pixels:
        return [[bg] * 3 for _ in range(3)]
    
    # Find bounding box
    min_r = min(r for r, c, v in pixels)
    max_r = max(r for r, c, v in pixels)
    min_c = min(c for r, c, v in pixels)
    max_c = max(c for r, c, v in pixels)
    
    # Create 3x3 output filled with background
    output = [[bg] * 3 for _ in range(3)]
    
    # Map each pixel to a cell in the 3x3 grid
    box_h = max_r - min_r
    box_w = max_c - min_c
    
    for r, c, v in pixels:
        # Normalize to [0, 1] range
        if box_h > 0:
            norm_r = (r - min_r) / box_h
        else:
            norm_r = 0.5
        
        if box_w > 0:
            norm_c = (c - min_c) / box_w
        else:
            norm_c = 0.5
        
        # Map to 3x3 grid cell (0, 1, or 2)
        grid_r = min(2, int(norm_r * 3))
        grid_c = min(2, int(norm_c * 3))
        
        output[grid_r][grid_c] = v
    
    return output
