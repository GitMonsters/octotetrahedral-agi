import copy

def transform(grid):
    """
    ARC puzzle 6aae56a9: Draw bracket shapes from colored points.
    
    For two points on the same row: Each point creates a bracket facing the other,
    meeting halfway between them.
    
    For a single point: Creates a bracket facing toward the grid center.
    
    Each bracket consists of:
    - A horizontal line from the point to the meeting column
    - A vertical line at the meeting column (2 cells up and down)
    - Corner extensions (1 cell toward the other point at top and bottom)
    """
    rows = len(grid)
    cols = len(grid[0])
    bg = grid[0][0]
    result = copy.deepcopy(grid)
    
    # Find non-background points
    points = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                points.append((r, c, grid[r][c]))
    
    def draw_bracket(row, col, color, vertical_col, corner_col):
        """Draw a bracket shape."""
        # Horizontal line from col to vertical_col
        start_c = min(col, vertical_col)
        end_c = max(col, vertical_col)
        for c in range(start_c, end_c + 1):
            result[row][c] = color
        
        # Vertical line at vertical_col (2 up, 2 down)
        for dr in [-2, -1, 1, 2]:
            if 0 <= row + dr < rows:
                result[row + dr][vertical_col] = color
        
        # Corner extensions at top and bottom
        for dr in [-2, 2]:
            if 0 <= row + dr < rows and 0 <= corner_col < cols:
                result[row + dr][corner_col] = color
    
    if len(points) == 2:
        # Two points case - they face each other
        points.sort(key=lambda x: x[1])  # Sort by column
        r1, c1, v1 = points[0]  # Left point
        r2, c2, v2 = points[1]  # Right point
        
        row = r1  # Assume same row
        dist = c2 - c1
        half = dist // 2
        
        # Left bracket (facing right)
        left_vert = c1 + half - 1
        left_corner = c1 + half
        draw_bracket(row, c1, v1, left_vert, left_corner)
        
        # Right bracket (facing left)
        right_vert = c2 - half + 1
        right_corner = c2 - half
        draw_bracket(row, c2, v2, right_vert, right_corner)
    
    elif len(points) == 1:
        # Single point case - bracket facing toward center
        r, c, v = points[0]
        center_c = cols // 2
        
        if c < center_c:
            # Point is on left, bracket faces right
            mid = (c + center_c) // 2
            draw_bracket(r, c, v, mid, mid + 1)
        else:
            # Point is on right, bracket faces left
            mid = (c + center_c) // 2
            draw_bracket(r, c, v, mid, mid - 1)
    
    return result
