def transform(grid):
    """
    Pattern: Tiling transformation based on detecting which edge has variation
    """
    from collections import Counter
    
    h = len(grid)
    w = len(grid[0])
    
    # Get statistics
    first_col = [grid[r][0] for r in range(h)]
    first_row = grid[0]
    last_row = grid[-1]
    
    first_col_unique = len(set(first_col))
    first_row_unique = len(set(first_row))
    last_row_unique = len(set(last_row))
    
    # Check if first column has significant variation
    # And first row is mostly uniform (background)
    first_row_counter = Counter(first_row)
    most_common_first_row = first_row_counter.most_common(1)[0][1]
    
    if first_col_unique >= h // 2 and most_common_first_row >= w - 2:
        # Horizontal tiling: keep col 0 and 1, then repeat first_col pattern
        result = []
        for r in range(h):
            row = [grid[r][0], grid[r][1]]
            # Repeat the first column pattern starting from col 2
            for c in range(2, w):
                row.append(first_col[(c - 2) % h])
            result.append(row)
        return result
    
    # Check if first row has variation and second row is uniform
    second_row = grid[1]
    if first_row_unique > 1 and len(set(second_row)) == 1:
        # Vertical tiling from first row: reverse and cycle
        reversed_row = list(reversed(first_row))
        
        result = [first_row[:], second_row[:]]
        # Tile the rest by cycling reversed_row
        for r in range(2, h):
            idx = (r - 2) % len(reversed_row)
            result.append([reversed_row[idx]] * w)
        return result
    
    # Check if last row has variation and rest is mostly uniform
    all_colors = Counter()
    for row in grid[:-1]:
        all_colors.update(row)
    
    if last_row_unique > 1 and len(all_colors) <= 2:
        # Find background color (most common)
        bg_color = all_colors.most_common(1)[0][0]
        
        # Vertical tiling from last row: reverse and cycle starting from last element
        reversed_row = list(reversed(last_row))
        n = len(reversed_row)
        
        result = []
        for r in range(h - 1):
            # Check if this row is already uniform and non-background - preserve it
            if len(set(grid[r])) == 1 and grid[r][0] != bg_color:
                result.append(grid[r][:])
            else:
                # Start from last element of reversed and cycle forward
                idx = (n - 1 + r) % n
                result.append([reversed_row[idx]] * w)
        result.append(last_row[:])
        return result
    
    # Default: return copy
    return [row[:] for row in grid]
