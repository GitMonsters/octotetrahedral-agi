from collections import Counter

def transform(input_grid):
    nrows = len(input_grid)
    ncols = len(input_grid[0])
    
    # Find background color (most common)
    bg = Counter(c for row in input_grid for c in row).most_common(1)[0][0]
    
    # Threshold: bottom 1/3 of rows become yellow
    threshold = nrows * 2 // 3
    
    # For each column, find the row of its single non-bg pixel
    row_pattern = []
    for j in range(ncols):
        pixel_row = 0
        for i in range(nrows):
            if input_grid[i][j] != bg:
                pixel_row = i
                break
        # Yellow(4) if in bottom third, Gray(5) otherwise
        row_pattern.append(4 if pixel_row >= threshold else 5)
    
    # Output: every row is the same pattern
    return [list(row_pattern) for _ in range(nrows)]
