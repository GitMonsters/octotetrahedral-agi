def transform(grid):
    """
    Pattern: A template pattern on one edge propagates across the grid,
    shifting by 1 at each marker (color 3) on a perpendicular edge.
    """
    import copy
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    all_vals = [v for row in grid for v in row]
    bg = Counter(all_vals).most_common(1)[0][0]
    
    # Find pattern and marker edges
    # Check each edge for pattern (non-bg, non-3) and markers (3)
    top_row = grid[0]
    bot_row = grid[h-1]
    left_col = [grid[r][0] for r in range(h)]
    right_col = [grid[r][w-1] for r in range(h)]
    
    def has_pattern(edge):
        return any(v != bg and v != 3 for v in edge)
    
    def has_markers(edge):
        return any(v == 3 for v in edge)
    
    def get_marker_positions(edge):
        return [i for i, v in enumerate(edge) if v == 3]
    
    pattern_edge = None
    pattern = None
    marker_edge = None
    markers = None
    orientation = None  # 'horizontal' or 'vertical'
    shift_dir = None  # +1 or -1
    
    # Determine pattern and marker locations
    if has_pattern(top_row) and has_markers(left_col):
        pattern_edge = 'top'
        pattern = top_row[:]
        markers = get_marker_positions(left_col)
        orientation = 'horizontal'
        shift_dir = 1  # shift right as we go down
    elif has_pattern(top_row) and has_markers(right_col):
        pattern_edge = 'top'
        pattern = top_row[:]
        markers = get_marker_positions(right_col)
        orientation = 'horizontal'
        shift_dir = -1  # shift left as we go down
    elif has_pattern(bot_row) and has_markers(left_col):
        pattern_edge = 'bottom'
        pattern = bot_row[:]
        markers = get_marker_positions(left_col)
        orientation = 'horizontal'
        shift_dir = 1  # shift right as we go up
    elif has_pattern(bot_row) and has_markers(right_col):
        pattern_edge = 'bottom'
        pattern = bot_row[:]
        markers = get_marker_positions(right_col)
        orientation = 'horizontal'
        shift_dir = -1  # shift left as we go up
    elif has_pattern(left_col) and has_markers(top_row):
        pattern_edge = 'left'
        pattern = left_col[:]
        markers = get_marker_positions(top_row)
        orientation = 'vertical'
        shift_dir = 1  # shift down as we go right
    elif has_pattern(left_col) and has_markers(bot_row):
        pattern_edge = 'left'
        pattern = left_col[:]
        markers = get_marker_positions(bot_row)
        orientation = 'vertical'
        shift_dir = -1  # shift up as we go right
    elif has_pattern(right_col) and has_markers(top_row):
        pattern_edge = 'right'
        pattern = right_col[:]
        markers = get_marker_positions(top_row)
        orientation = 'vertical'
        shift_dir = -1  # shift up as we go left
    elif has_pattern(right_col) and has_markers(bot_row):
        pattern_edge = 'right'
        pattern = right_col[:]
        markers = get_marker_positions(bot_row)
        orientation = 'vertical'
        shift_dir = 1  # shift down as we go left
    
    # If no pattern found, return original
    if pattern is None or not markers:
        return grid
    
    result = [[bg for _ in range(w)] for _ in range(h)]
    
    # Keep markers in place
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 3:
                result[r][c] = 3
    
    if orientation == 'horizontal':
        # Pattern is a row, markers are in a column
        markers_sorted = sorted(markers)
        
        for r in range(h):
            # Count how many markers we've passed
            if pattern_edge == 'top':
                shift_count = sum(1 for m in markers_sorted if m <= r)
            else:  # bottom
                shift_count = sum(1 for m in markers_sorted if m >= r)
            
            shift = shift_count * shift_dir
            
            for c in range(w):
                src_c = c - shift
                if 0 <= src_c < w:
                    if pattern[src_c] != bg and pattern[src_c] != 3:
                        result[r][c] = pattern[src_c]
    
    else:  # vertical
        # Pattern is a column, markers are in a row
        markers_sorted = sorted(markers)
        
        for c in range(w):
            # Count how many markers we've passed
            if pattern_edge == 'left':
                shift_count = sum(1 for m in markers_sorted if m <= c)
            else:  # right
                shift_count = sum(1 for m in markers_sorted if m >= c)
            
            shift = shift_count * shift_dir
            
            for r in range(h):
                src_r = r - shift
                if 0 <= src_r < h:
                    if pattern[src_r] != bg and pattern[src_r] != 3:
                        result[r][c] = pattern[src_r]
    
    return result
