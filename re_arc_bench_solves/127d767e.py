"""
ARC Puzzle 127d767e Solver

Pattern: The input has a separator (full row & column of same color) dividing it.
- Top-left 2x2 contains marker colors for 4 quadrants
- Data area (below and right of separator) contains a pattern on a background
- Output: data area with pattern pixels colored by quadrant marker colors
"""
from collections import Counter


def transform(grid):
    rows, cols = len(grid), len(grid[0])
    
    # Find separator row (full row of same color)
    sep_row = None
    sep_color = None
    for r in range(rows):
        if len(set(grid[r])) == 1:
            sep_row = r
            sep_color = grid[r][0]
            break
    
    # Find separator column (full column of same color)
    sep_col = None
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1:
            sep_col = c
            break
    
    # Get marker colors from 2x2 region before separator
    top_left = grid[0][0]
    top_right = grid[0][1] if sep_col > 1 else grid[0][0]
    bottom_left = grid[1][0]
    bottom_right = grid[1][1] if sep_col > 1 else grid[1][0]
    
    # Data area starts after separator
    data_start_row = sep_row + 1
    data_start_col = sep_col + 1
    
    data_rows = rows - data_start_row
    data_cols = cols - data_start_col
    
    # Background color is from the region left of data area (same rows, before sep_col)
    background_color = grid[data_start_row][0]
    
    # Pattern color is the non-background color in data area
    data_values = [grid[r][c] for r in range(data_start_row, rows) 
                   for c in range(data_start_col, cols)]
    counts = Counter(data_values)
    
    pattern_color = None
    for color, _ in counts.most_common():
        if color != background_color:
            pattern_color = color
            break
    
    if pattern_color is None:
        pattern_color = background_color
    
    # Quadrant midpoints
    mid_row = data_rows // 2
    mid_col = data_cols // 2
    
    # Build output: pattern pixels get marker colors based on quadrant
    output = []
    for r in range(data_rows):
        row = []
        for c in range(data_cols):
            val = grid[data_start_row + r][data_start_col + c]
            
            in_top = r < mid_row
            in_left = c < mid_col
            
            if val == pattern_color:
                if in_top and in_left:
                    row.append(top_left)
                elif in_top and not in_left:
                    row.append(top_right)
                elif not in_top and in_left:
                    row.append(bottom_left)
                else:
                    row.append(bottom_right)
            else:
                row.append(val)
        output.append(row)
    
    return output
