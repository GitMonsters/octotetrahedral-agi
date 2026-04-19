from collections import Counter

def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find background color
    flat = [v for row in input_grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background cells
    specials = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != bg:
                specials.append((r, c, input_grid[r][c]))
    
    # Sort by column DESCENDING (tiebreak by row ascending)
    specials.sort(key=lambda x: (-x[1], x[0]))
    
    # Compute normalized spread
    spec_rows = [r for r, c, v in specials]
    spec_cols = [c for r, c, v in specials]
    row_spread = (max(spec_rows) - min(spec_rows)) / rows
    col_spread = (max(spec_cols) - min(spec_cols)) / cols
    
    # Create 3x3 output filled with background
    output = [[bg] * 3 for _ in range(3)]
    
    # Determine fill order
    if col_spread >= row_spread:
        # Horizontal: fill rows from bottom, left to right
        positions = [(2,0),(2,1),(2,2),(1,0),(1,1),(1,2),(0,0),(0,1),(0,2)]
    else:
        # Vertical: fill columns from bottom, left to right
        positions = [(2,0),(1,0),(0,0),(2,1),(1,1),(0,1),(2,2),(1,2),(0,2)]
    
    # Place values
    for i, (r, c, v) in enumerate(specials):
        if i < len(positions):
            pr, pc = positions[i]
            output[pr][pc] = v
    
    return output
