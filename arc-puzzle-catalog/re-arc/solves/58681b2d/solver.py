from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = [row[:] for row in input_grid]
    
    for c in range(cols):
        col_vals = [input_grid[r][c] for r in range(rows)]
        most_common = Counter(col_vals).most_common(1)[0][0]
        for r in range(rows):
            output[r][c] = most_common
    
    return output
