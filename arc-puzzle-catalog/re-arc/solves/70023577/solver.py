def get_background_color(grid):
    """Find the most frequent color in the grid."""
    from collections import Counter
    all_cells = [cell for row in grid for cell in row]
    return Counter(all_cells).most_common(1)[0][0]


def find_non_background_cells(grid, bg):
    """Find all non-background cell positions and their values."""
    non_bg = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val != bg:
                non_bg.append((r, c, val))
    return non_bg


def is_seed_row(non_bg_cells):
    """Check if all non-background cells are in a single row."""
    if not non_bg_cells:
        return False
    rows = set(r for r, c, v in non_bg_cells)
    return len(rows) == 1


def is_seed_column(non_bg_cells):
    """Check if all non-background cells are in a single column."""
    if not non_bg_cells:
        return False
    cols = set(c for r, c, v in non_bg_cells)
    return len(cols) == 1


def dilate_line(line, bg):
    """
    Dilate a line by expanding non-background cells to adjacent positions.
    For each non-background cell, place the same color at adjacent positions
    (left/right for row, up/down for column).
    """
    expanded = [bg] * len(line)
    for i, val in enumerate(line):
        if val != bg:
            # Place at adjacent positions
            if i > 0:
                expanded[i - 1] = val
            if i < len(line) - 1:
                expanded[i + 1] = val
    return expanded


def transform(grid):
    """
    Transform the grid according to the seed line expansion rule.
    """
    # Find background color
    bg = get_background_color(grid)
    
    # Find non-background cells
    non_bg = find_non_background_cells(grid, bg)
    
    # If no non-background cells, return a copy unchanged
    if not non_bg:
        return [row[:] for row in grid]
    
    height = len(grid)
    width = len(grid[0])
    
    # Determine if seed is a row or column
    if is_seed_row(non_bg):
        # Seed is a row
        seed_row_idx = non_bg[0][0]
        base = grid[seed_row_idx][:]
        expanded = dilate_line(base, bg)
        
        # Determine direction (from top or bottom)
        is_bottom = seed_row_idx == height - 1
        is_top = seed_row_idx == 0
        
        # Fill the grid
        result = [[bg] * width for _ in range(height)]
        
        if is_bottom:
            # Alternate upward from bottom
            for r in range(height):
                dist = height - 1 - r
                if dist % 2 == 0:
                    result[r] = base[:]
                else:
                    result[r] = expanded[:]
        elif is_top:
            # Alternate downward from top
            for r in range(height):
                dist = r
                if dist % 2 == 0:
                    result[r] = base[:]
                else:
                    result[r] = expanded[:]
        else:
            # Seed row is in the middle - alternate both directions
            # First, handle distance from seed row
            for r in range(height):
                dist = abs(r - seed_row_idx)
                if dist % 2 == 0:
                    result[r] = base[:]
                else:
                    result[r] = expanded[:]
        
        return result
    
    elif is_seed_column(non_bg):
        # Seed is a column
        seed_col_idx = non_bg[0][1]
        base = [grid[r][seed_col_idx] for r in range(height)]
        expanded = dilate_line(base, bg)
        
        # Determine direction (from left or right)
        is_right = seed_col_idx == width - 1
        is_left = seed_col_idx == 0
        
        # Fill the grid
        result = [[bg] * width for _ in range(height)]
        
        if is_right:
            # Alternate leftward from right
            for c in range(width):
                dist = width - 1 - c
                if dist % 2 == 0:
                    for r in range(height):
                        result[r][c] = base[r]
                else:
                    for r in range(height):
                        result[r][c] = expanded[r]
        elif is_left:
            # Alternate rightward from left
            for c in range(width):
                dist = c
                if dist % 2 == 0:
                    for r in range(height):
                        result[r][c] = base[r]
                else:
                    for r in range(height):
                        result[r][c] = expanded[r]
        else:
            # Seed column is in the middle - alternate both directions
            for c in range(width):
                dist = abs(c - seed_col_idx)
                if dist % 2 == 0:
                    for r in range(height):
                        result[r][c] = base[r]
                else:
                    for r in range(height):
                        result[r][c] = expanded[r]
        
        return result
    
    else:
        # Non-background cells don't form a single row or column
        # This shouldn't happen according to the rule, return unchanged
        return [row[:] for row in grid]
