def transform(input_grid):
    from collections import Counter
    
    grid = [row[:] for row in input_grid]
    H = len(grid)
    W = len(grid[0])
    
    # Find background color (most common)
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find all non-background pixels and track which columns have pixels
    pixels = []
    pixel_cols = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                pixels.append((r, c, grid[r][c]))
                pixel_cols.add(c)
    
    output = [row[:] for row in input_grid]
    
    # Each pixel creates a downward trail alternating between its color and 6
    for (pr, pc, pv) in pixels:
        for r in range(pr + 1, H):
            dist = r - pr
            val = 6 if dist % 2 == 1 else pv
            output[r][pc] = val
    
    # Pixel at second-to-last row with both adjacent columns free:
    # creates an upward trail of 6 at col+1, every other row
    for (pr, pc, pv) in pixels:
        if pr == H - 2:
            col_left_free = pc - 1 >= 0 and (pc - 1) not in pixel_cols
            col_right_free = pc + 1 < W and (pc + 1) not in pixel_cols
            if col_left_free and col_right_free:
                for r in range(pr, -1, -2):
                    if output[r][pc + 1] == bg:
                        output[r][pc + 1] = 6
    
    return output
