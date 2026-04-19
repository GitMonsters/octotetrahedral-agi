from collections import Counter

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Count colors
    counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    colors = sorted(counts.items(), key=lambda x: -x[1])
    
    if len(colors) < 3:
        # 2-color grid: no markers, return as-is
        return [row[:] for row in grid]
    
    # 3 colors: most common is center, least common is marker
    center_color = colors[0][0]
    border_color = colors[1][0]
    marker_color = colors[2][0]
    
    # Find stripe boundaries from baseline row
    baseline = Counter(tuple(r) for r in grid).most_common(1)[0][0]
    
    # Find left border width
    left_width = 0
    for c in range(cols):
        if baseline[c] == border_color:
            left_width += 1
        else:
            break
    
    # Find right border width
    right_width = 0
    for c in range(cols-1, -1, -1):
        if baseline[c] == border_color:
            right_width += 1
        else:
            break
    
    center_start = left_width
    center_end = cols - right_width  # exclusive
    
    output = []
    for r in range(rows):
        row = grid[r]
        
        # Count markers in left border
        n_left = sum(1 for c in range(center_start) if row[c] == marker_color)
        
        # Count markers in right border
        n_right = sum(1 for c in range(center_end, cols) if row[c] == marker_color)
        
        # Build output row
        new_left_width = left_width - n_left
        new_right_width = right_width - n_right
        new_center_width = cols - new_left_width - new_right_width
        
        out_row = ([border_color] * new_left_width + 
                   [center_color] * new_center_width + 
                   [border_color] * new_right_width)
        output.append(out_row)
    
    return output

