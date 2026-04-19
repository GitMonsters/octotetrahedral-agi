def transform(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Count magenta (6) cells — this is the scaling factor N
    N = sum(1 for r in input_grid for c in r if c == 6)
    
    # Count non-magenta (background) cells
    total_bg = rows * cols - N
    
    # Distribute bg cells across N meta-rows (right-aligned staircase)
    base = total_bg // N
    extra = total_bg % N
    
    # Create output filled with magenta (6)
    out_rows = rows * N
    out_cols = cols * N
    output = [[6] * out_cols for _ in range(out_rows)]
    
    # Place input copies in the meta-grid
    for meta_r in range(N):
        width = base + (1 if meta_r >= N - extra else 0)
        for w in range(width):
            meta_c = N - width + w
            # Copy input into block (meta_r, meta_c)
            for r in range(rows):
                for c in range(cols):
                    output[meta_r * rows + r][meta_c * cols + c] = input_grid[r][c]
    
    return output
