from collections import Counter

def transform(grid):
    """
    Project scattered pixels toward the main rectangular shape.
    - Pixels above/below shape project to row adjacent to shape, keeping column
    - Pixels left/right of shape project to column adjacent to shape, keeping row
    - Diagonal pixels project to the nearest corner of the shape's bounding box
    """
    rows, cols = len(grid), len(grid[0])
    
    # Find background (most common color)
    flat = [c for row in grid for c in row]
    color_counts = Counter(flat)
    bg = color_counts.most_common(1)[0][0]
    
    # Find other colors - shape (more pixels) and scatter (fewer pixels)
    other_colors = [c for c in color_counts if c != bg]
    if len(other_colors) < 2:
        return grid
    
    shape_color = max(other_colors, key=lambda x: color_counts[x])
    scatter_color = min(other_colors, key=lambda x: color_counts[x])
    
    # Get shape bounding box
    shape_pos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color]
    min_r = min(p[0] for p in shape_pos)
    max_r = max(p[0] for p in shape_pos)
    min_c = min(p[1] for p in shape_pos)
    max_c = max(p[1] for p in shape_pos)
    
    # Find scattered pixels
    scatter_pos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == scatter_color]
    
    # Create output grid - start with all background
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    # Copy shape to output
    for r, c in shape_pos:
        output[r][c] = shape_color
    
    # Project scattered pixels toward shape
    for r, c in scatter_pos:
        above = r < min_r
        below = r > max_r
        left = c < min_c
        right = c > max_c
        
        if above and left:
            new_r, new_c = min_r - 1, min_c - 1
        elif above and right:
            new_r, new_c = min_r - 1, max_c + 1
        elif below and left:
            new_r, new_c = max_r + 1, min_c - 1
        elif below and right:
            new_r, new_c = max_r + 1, max_c + 1
        elif above:
            new_r, new_c = min_r - 1, c
        elif below:
            new_r, new_c = max_r + 1, c
        elif left:
            new_r, new_c = r, min_c - 1
        elif right:
            new_r, new_c = r, max_c + 1
        else:
            new_r, new_c = r, c
        
        if 0 <= new_r < rows and 0 <= new_c < cols:
            output[new_r][new_c] = scatter_color
    
    return output
