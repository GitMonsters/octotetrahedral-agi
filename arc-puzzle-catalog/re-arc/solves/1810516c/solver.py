import numpy as np


def transform(grid):
    """
    Transform filled rectangles to hollow rectangles with alternating interior.
    
    Rules:
    - Rectangles with width=3: hollow center column at odd row offsets
    - Rectangles with height=3: hollow center row at odd column offsets
    - Background color is the most common color in the grid
    """
    grid = np.array(grid, dtype=int)
    output = grid.copy()
    height, width = grid.shape
    
    # Find background color (most common)
    flat = grid.flatten()
    background = np.bincount(flat).argmax()
    
    # Find all rectangular regions
    visited = np.zeros_like(grid, dtype=bool)
    rectangles = []
    
    for r in range(height):
        for c in range(width):
            if not visited[r, c] and grid[r, c] != background:
                color = grid[r, c]
                
                # Find rectangular bounds
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                # Expand right
                while max_c + 1 < width and grid[r, max_c + 1] == color:
                    max_c += 1
                
                # Expand down
                found = True
                while max_r + 1 < height and found:
                    for cc in range(min_c, max_c + 1):
                        if grid[max_r + 1, cc] != color:
                            found = False
                            break
                    if found:
                        max_r += 1
                
                # Verify it's a solid rectangle
                is_rect = True
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        if grid[rr, cc] != color:
                            is_rect = False
                            break
                    if not is_rect:
                        break
                
                if is_rect:
                    rectangles.append((min_r, max_r, min_c, max_c, color))
                    visited[min_r:max_r+1, min_c:max_c+1] = True
    
    # Apply hollowing based on dimensions
    for min_r, max_r, min_c, max_c, color in rectangles:
        rect_height = max_r - min_r + 1
        rect_width = max_c - min_c + 1
        
        if rect_width == 3:
            # Hollow center column at odd row offsets
            center_c = min_c + 1
            for r in range(min_r, max_r + 1):
                interior_r = r - min_r
                if interior_r % 2 == 1:
                    output[r, center_c] = background
        
        elif rect_height == 3:
            # Hollow center row at odd column offsets
            center_r = min_r + 1
            for c in range(min_c, max_c + 1):
                interior_c = c - min_c
                if interior_c % 2 == 1:
                    output[center_r, c] = background
    
    return output.tolist()
