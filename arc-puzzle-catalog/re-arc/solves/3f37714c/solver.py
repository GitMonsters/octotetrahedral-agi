from collections import Counter

def transform(grid):
    """
    Pattern: Markers in a column indicate rows to fill with checkerboard.
    - Consecutive markers of same color (gap <= 4) form a group
    - Fill from (min_row - 1) to (max_row + 1) with checkerboard
    - Marker rows: color at even column positions
    - Non-marker rows: color at odd column positions
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # Copy input to output
    output = [row[:] for row in grid]
    
    # Determine background color (most common)
    all_vals = [v for row in grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Find all markers (non-background values)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                markers.append((r, c, grid[r][c]))
    
    if not markers:
        return output
    
    # Group markers by color
    color_markers = {}
    for r, c, color in markers:
        if color not in color_markers:
            color_markers[color] = []
        color_markers[color].append((r, c))
    
    # Process each color group
    for color, positions in color_markers.items():
        rows_with_marker = sorted([r for r, c in positions])
        
        # Group consecutive markers (gap <= 4 rows)
        groups = []
        current_group = [rows_with_marker[0]]
        for i in range(1, len(rows_with_marker)):
            if rows_with_marker[i] - rows_with_marker[i-1] <= 4:
                current_group.append(rows_with_marker[i])
            else:
                groups.append(current_group)
                current_group = [rows_with_marker[i]]
        groups.append(current_group)
        
        # Fill checkerboard for each group
        for group in groups:
            min_r = max(0, min(group) - 1)
            max_r = min(rows - 1, max(group) + 1)
            
            marker_set = set(group)
            
            for r in range(min_r, max_r + 1):
                for c in range(cols):
                    # Marker rows: color at even column positions
                    # Non-marker rows: color at odd column positions
                    if r in marker_set:
                        if c % 2 == 0:
                            output[r][c] = color
                        else:
                            output[r][c] = bg
                    else:
                        if c % 2 == 1:
                            output[r][c] = color
                        else:
                            output[r][c] = bg
    
    return output
