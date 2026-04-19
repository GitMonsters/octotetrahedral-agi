def transform(grid):
    from collections import Counter
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    all_vals = [v for row in grid for v in row]
    bg_color = Counter(all_vals).most_common(1)[0][0]
    
    # Grid is divided into 3 horizontal bands
    band_size = rows // 3
    
    # Band determines output color: top->1(blue), middle->7(orange), bottom->3(green)
    band_to_color = {0: 1, 1: 7, 2: 3}
    
    output = [[0] * cols for _ in range(rows)]
    
    for c in range(cols):
        # Find the dot row in this column
        dot_row = None
        for r in range(rows):
            if grid[r][c] != bg_color:
                dot_row = r
                break
        
        if dot_row is not None:
            band = min(dot_row // band_size, 2) if band_size > 0 else 0
            color = band_to_color[band]
        else:
            color = bg_color
        
        # Fill entire column with that color
        for r in range(rows):
            output[r][c] = color
    
    return output
