def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the background color (most common) and marker color (least common, not background)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1
    
    # Background is most common, marker is the other color
    sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
    background_color = sorted_colors[0][0]
    marker_color = sorted_colors[1][0] if len(sorted_colors) > 1 else background_color
    
    # For each row, find the column position of the marker (non-background cell)
    row_marker_positions = []
    for r in range(rows):
        marker_col = None
        for c in range(cols):
            if grid[r][c] != background_color:
                marker_col = c
                break
        row_marker_positions.append(marker_col)
    
    # The output colors seem to be based on the marker position
    # Looking at the examples more carefully:
    # - The output fills each row with a single color
    # - The color depends on where the marker is in that row
    
    # Let's analyze the mapping:
    # In example 1: background=7, marker=0
    # Positions and output colors vary
    
    # In example 3 (simpler):
    # Row 0: marker at col 5 (rightmost) -> output 1
    # Row 1: marker at col 1 (left) -> output 6
    # Row 2: marker at col 1 (left) -> output 6
    # Row 3: marker at col 3 (middle) -> output 7
    # Row 4: marker at col 4 (middle-right) -> output 1
    # Row 5: marker at col 1 (left) -> output 6
    
    # It seems like the position determines the output color
    # Let's divide the grid into 3 zones: left, middle, right
    # Left -> 6, Middle -> 7 (background), Right -> 1
    
    # Actually looking at example 4 which has 24 columns and marker=1, background=7:
    # The output uses colors 6, 7, 1
    
    # Let me re-examine: the three colors used in output are always 1, 6, 7
    # And which one is used depends on the marker column position
    
    # Dividing into thirds:
    third = cols / 3
    
    result = []
    for r in range(rows):
        marker_col = row_marker_positions[r]
        if marker_col is None:
            # No marker, keep background
            result.append([background_color] * cols)
        else:
            # Determine which third the marker is in
            if marker_col < third:
                # Left third -> color 6
                fill_color = 6
            elif marker_col < 2 * third:
                # Middle third -> color 7
                fill_color = 7
            else:
                # Right third -> color 1
                fill_color = 1
            result.append([fill_color] * cols)
    
    return result