import numpy as np
from scipy import ndimage

def transform(grid):
    """
    ARC puzzle 3ec0681f: Project noise lines through a solid rectangle.
    
    Pattern: Find a solid rectangular region. For each row/column passing through it,
    extend any noise pixels on one side to fill up to the rectangle edge,
    and from the opposite rectangle edge to the farthest noise on that side.
    """
    grid = np.array(grid)
    h, w = grid.shape
    result = grid.copy()
    
    colors, counts = np.unique(grid, return_counts=True)
    
    # Find the solid rectangle: largest connected component that exactly fills its bounding box
    rect_color = None
    rect_bounds = None
    max_rect_size = 0
    
    for c in colors:
        mask = grid == c
        labeled, num = ndimage.label(mask)
        
        for i in range(1, num + 1):
            comp = labeled == i
            rows, cols = np.where(comp)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            comp_size = np.sum(comp)
            rect_size = (r_max - r_min + 1) * (c_max - c_min + 1)
            
            # Must be a solid rectangle (component fills bounding box exactly)
            # and at least 2x2
            if comp_size == rect_size and comp_size > max_rect_size and (r_max - r_min + 1) >= 2 and (c_max - c_min + 1) >= 2:
                max_rect_size = comp_size
                rect_color = c
                rect_bounds = (r_min, r_max, c_min, c_max)
    
    if rect_color is None:
        return result.tolist()
    
    r_min, r_max, c_min, c_max = rect_bounds
    
    # Determine background vs noise
    # If only 2 colors: rect_color serves as both rect AND noise; other color is background
    # If 3+ colors: background is most common outside rect, noise is the remaining color
    
    if len(colors) == 2:
        bg_color = [c for c in colors if c != rect_color][0]
        noise_color = rect_color  # Same color used for noise
    else:
        outside_mask = np.ones_like(grid, dtype=bool)
        outside_mask[r_min:r_max+1, c_min:c_max+1] = False
        outside_grid = grid[outside_mask]
        bg_colors, bg_counts = np.unique(outside_grid, return_counts=True)
        bg_color = bg_colors[np.argmax(bg_counts)]
        noise_colors = [c for c in colors if c != rect_color and c != bg_color]
        if not noise_colors:
            return result.tolist()
        noise_color = noise_colors[0]
    
    # For each row that passes through rectangle:
    for r in range(r_min, r_max + 1):
        # Find noise positions outside the rectangle in this row
        noise_left = [c for c in range(c_min) if grid[r, c] == noise_color]
        noise_right = [c for c in range(c_max + 1, w) if grid[r, c] == noise_color]
        
        if noise_left:
            leftmost = min(noise_left)
            # Fill from leftmost noise to rect left edge
            for c in range(leftmost, c_min):
                if result[r, c] == bg_color:
                    result[r, c] = noise_color
        
        if noise_right:
            rightmost = max(noise_right)
            # Fill from rect right edge to rightmost noise
            for c in range(c_max + 1, rightmost + 1):
                if result[r, c] == bg_color:
                    result[r, c] = noise_color
    
    # For each column that passes through rectangle:
    for c in range(c_min, c_max + 1):
        noise_above = [r for r in range(r_min) if grid[r, c] == noise_color]
        noise_below = [r for r in range(r_max + 1, h) if grid[r, c] == noise_color]
        
        if noise_above:
            topmost = min(noise_above)
            for r in range(topmost, r_min):
                if result[r, c] == bg_color:
                    result[r, c] = noise_color
        
        if noise_below:
            bottommost = max(noise_below)
            for r in range(r_max + 1, bottommost + 1):
                if result[r, c] == bg_color:
                    result[r, c] = noise_color
    
    return result.tolist()
