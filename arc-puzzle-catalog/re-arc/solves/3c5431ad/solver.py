import numpy as np

def find_maximal_solid_rectangles(grid, color, min_dim=2):
    """Find maximal axis-aligned solid rectangles of a specific color.
    
    Args:
        grid: Input grid as numpy array
        color: Target color to find rectangles of
        min_dim: Minimum dimension (both height and width must be >= min_dim)
    
    Returns:
        List of (r_start, r_end, c_start, c_end, size) tuples
    """
    h, w = grid.shape
    rectangles = []
    checked = set()
    
    for r_start in range(h):
        for c_start in range(w):
            if grid[r_start, c_start] != color:
                continue
            
            # Find maximum width at this starting row
            max_width = 0
            for c_end in range(c_start, w):
                if grid[r_start, c_end] == color:
                    max_width = c_end - c_start + 1
                else:
                    break
            
            if max_width == 0 or max_width < min_dim:
                continue
            
            # Find maximum height with this width
            max_height = 0
            for r_end in range(r_start, h):
                if np.all(grid[r_end, c_start:c_start+max_width] == color):
                    max_height = r_end - r_start + 1
                else:
                    break
            
            if max_height < min_dim:
                continue
            
            # Avoid duplicate rectangles
            rect_key = (r_start, r_start + max_height - 1, c_start, c_start + max_width - 1)
            if rect_key in checked:
                continue
            
            size = max_height * max_width
            rectangles.append((*rect_key, size))
            checked.add(rect_key)
    
    return rectangles


def transform(grid):
    """Extract clean solid rectangles from noisy input grid.
    
    The transformation identifies the most common non-2 color as the background,
    then creates an output grid that is either:
    1. Filled with the background color (if bg fills > 50% of input)
    2. Filled with 2s (if bg fills <= 50% of input)
    
    Solid rectangular regions are extracted and placed in their original positions.
    """
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    
    # Find the background color (most common non-2 value)
    non2 = grid[grid != 2]
    if len(non2) == 0:
        return grid.tolist()
    
    bg_vals, bg_counts = np.unique(non2, return_counts=True)
    bg_color = bg_vals[np.argmax(bg_counts)]
    
    # Check how much of the grid the background fills
    bg_fill = np.sum(grid == bg_color) / (h * w)
    
    if bg_fill > 0.5:
        # Strategy: Fill output with background, place other solid rectangles
        output = np.full((h, w), bg_color, dtype=int)
        
        all_colors = set(np.unique(grid)) - {2, bg_color}
        for color in all_colors:
            rectangles = find_maximal_solid_rectangles(grid, color, min_dim=2)
            for r_start, r_end, c_start, c_end, size in rectangles:
                output[r_start:r_end+1, c_start:c_end+1] = color
    else:
        # Strategy: Fill output with 2s, place background solid rectangles
        output = np.full((h, w), 2, dtype=int)
        
        rectangles = find_maximal_solid_rectangles(grid, bg_color, min_dim=2)
        for r_start, r_end, c_start, c_end, size in rectangles:
            output[r_start:r_end+1, c_start:c_end+1] = bg_color
    
    return output.tolist()
