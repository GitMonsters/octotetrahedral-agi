from collections import Counter


def transform(grid):
    """
    Transform ARC task 5d72370d:
    1. Determine background as the most frequent color.
    2. Find which border side contains the most non-background cells.
    3. Rotate grid so that chosen border becomes the right edge.
    4. On rotated grid, marker color = most common non-background on right edge.
    5. For each row whose rightmost cell is marker:
       - Find anchor (first non-bg, non-marker cell before last column).
       - If no anchor: fill whole row with marker.
       - If anchor at col c: fill from c+1 to end with marker, 
         fill next row from 0 to c+1 with marker.
    6. Rotate back to original orientation.
    """
    grid = [list(row) for row in grid]
    rows, cols = len(grid), len(grid[0])
    
    # Step 1: Determine background (most frequent color)
    all_cells = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_cells).most_common(1)[0][0]
    
    # Step 2: Count non-background cells on each border
    top_count = sum(1 for c in range(cols) if grid[0][c] != background)
    bottom_count = sum(1 for c in range(cols) if grid[rows-1][c] != background)
    left_count = sum(1 for r in range(rows) if grid[r][0] != background)
    right_count = sum(1 for r in range(rows) if grid[r][cols-1] != background)
    
    border_counts = {
        'top': top_count,
        'bottom': bottom_count,
        'left': left_count,
        'right': right_count
    }
    chosen_border = max(border_counts, key=border_counts.get)
    
    # Step 3: Rotate so chosen border becomes right edge
    # top->1 clockwise turn, left->2, bottom->3, right->0
    rotations = {'top': 1, 'left': 2, 'bottom': 3, 'right': 0}
    num_rotations = rotations[chosen_border]
    
    rotated = rotate_cw(grid, num_rotations)
    
    # Step 4: Marker color = most common non-background on right edge
    r_rows, r_cols = len(rotated), len(rotated[0])
    right_edge_colors = [rotated[r][r_cols-1] for r in range(r_rows) 
                         if rotated[r][r_cols-1] != background]
    marker = Counter(right_edge_colors).most_common(1)[0][0]
    
    # Step 5: Process rows whose rightmost cell is marker
    for r in range(r_rows):
        if rotated[r][r_cols-1] != marker:
            continue
        
        # Find anchor: first cell before last column that is neither bg nor marker
        anchor_col = None
        for c in range(r_cols - 1):
            if rotated[r][c] != background and rotated[r][c] != marker:
                anchor_col = c
                break
        
        if anchor_col is None:
            # No anchor: fill whole row with marker
            for c in range(r_cols):
                rotated[r][c] = marker
        else:
            # Anchor at column c: fill from c+1 through last column with marker
            for c in range(anchor_col + 1, r_cols):
                rotated[r][c] = marker
            # Fill next row (if any) from 0 through c+1 with marker
            if r + 1 < r_rows:
                for c in range(anchor_col + 2):
                    rotated[r + 1][c] = marker
    
    # Step 6: Rotate back to original orientation
    back_rotations = (4 - num_rotations) % 4
    result = rotate_cw(rotated, back_rotations)
    
    return result


def rotate_cw(grid, times):
    """Rotate grid clockwise 'times' times (each 90 degrees)."""
    result = [list(row) for row in grid]
    for _ in range(times % 4):
        rows, cols = len(result), len(result[0])
        result = [[result[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]
    return result
