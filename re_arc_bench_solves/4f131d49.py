"""
ARC Puzzle 4f131d49 Solver

Pattern: Markers on edges indicate lines to draw
- Markers on top/bottom row → vertical lines through those columns
- Markers on left/right column → horizontal lines through those rows
- Multiple markers with consistent spacing → extrapolate the pattern
"""

def transform(grid):
    import copy
    
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    bg_color = max(color_counts, key=color_counts.get)
    
    # Find all marker pixels (non-background)
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg_color:
                markers.append((r, c, grid[r][c]))
    
    if not markers:
        return grid
    
    # Determine if markers are on horizontal edge (top/bottom row) or vertical edge (left/right col)
    # Check for vertical edge markers (leftmost or rightmost column)
    left_col_markers = [(r, c, color) for r, c, color in markers if c == 0]
    right_col_markers = [(r, c, color) for r, c, color in markers if c == cols - 1]
    
    # Check for horizontal edge markers (top or bottom row)
    top_row_markers = [(r, c, color) for r, c, color in markers if r == 0]
    bottom_row_markers = [(r, c, color) for r, c, color in markers if r == rows - 1]
    
    output = copy.deepcopy(grid)
    
    # Process vertical edge markers → draw horizontal lines
    for edge_markers in [left_col_markers, right_col_markers]:
        if edge_markers:
            # Sort by row
            edge_markers.sort(key=lambda x: x[0])
            
            # Get positions and colors
            positions = [m[0] for m in edge_markers]
            colors = [m[2] for m in edge_markers]
            
            # Draw lines at marker positions
            line_positions = list(zip(positions, colors))
            
            # If multiple markers, extrapolate pattern
            if len(positions) >= 2:
                gap = positions[1] - positions[0]
                # Extrapolate forward
                next_pos = positions[-1] + gap
                color_idx = 0  # Alternate colors starting from first
                while 0 <= next_pos < rows:
                    line_positions.append((next_pos, colors[color_idx % len(colors)]))
                    next_pos += gap
                    color_idx += 1
                # Extrapolate backward
                prev_pos = positions[0] - gap
                color_idx = len(colors) - 1
                while 0 <= prev_pos < rows:
                    line_positions.insert(0, (prev_pos, colors[color_idx % len(colors)]))
                    prev_pos -= gap
                    color_idx -= 1
            
            # Draw horizontal lines
            for pos, color in line_positions:
                if 0 <= pos < rows:
                    for c in range(cols):
                        output[pos][c] = color
    
    # Process horizontal edge markers → draw vertical lines
    for edge_markers in [top_row_markers, bottom_row_markers]:
        if edge_markers:
            # Sort by column
            edge_markers.sort(key=lambda x: x[1])
            
            # Get positions and colors
            positions = [m[1] for m in edge_markers]
            colors = [m[2] for m in edge_markers]
            
            # Draw lines at marker positions
            line_positions = list(zip(positions, colors))
            
            # If multiple markers, extrapolate pattern
            if len(positions) >= 2:
                gap = positions[1] - positions[0]
                # Extrapolate forward
                next_pos = positions[-1] + gap
                color_idx = 0
                while 0 <= next_pos < cols:
                    line_positions.append((next_pos, colors[color_idx % len(colors)]))
                    next_pos += gap
                    color_idx += 1
                # Extrapolate backward
                prev_pos = positions[0] - gap
                color_idx = len(colors) - 1
                while 0 <= prev_pos < cols:
                    line_positions.insert(0, (prev_pos, colors[color_idx % len(colors)]))
                    prev_pos -= gap
                    color_idx -= 1
            
            # Draw vertical lines
            for pos, color in line_positions:
                if 0 <= pos < cols:
                    for r in range(rows):
                        output[r][pos] = color
    
    return output
