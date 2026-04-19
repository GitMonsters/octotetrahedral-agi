def transform(grid):
    """
    ARC puzzle 60795d0c solver.
    
    Pattern:
    1. Find full row/column dividers (entire row or column is one non-background color)
    2. Scattered pixels matching a divider color get projected onto the adjacent 
       row/column of that divider (on the same side as the pixel)
    3. All other noise pixels are cleared to background
    """
    import copy
    
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    all_pixels = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(all_pixels).most_common(1)[0][0]
    
    # Find horizontal dividers (full rows of same non-bg color)
    h_dividers = {}  # row_idx -> color
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != bg:
            h_dividers[r] = grid[r][0]
    
    # Find vertical dividers (full columns of same non-bg color)
    v_dividers = {}  # col_idx -> color
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] != bg:
            v_dividers[c] = col_vals[0]
    
    divider_colors = set(h_dividers.values()) | set(v_dividers.values())
    
    # Create output grid (start with background)
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    
    # Restore divider lines
    for r, color in h_dividers.items():
        for c in range(cols):
            output[r][c] = color
    for c, color in v_dividers.items():
        for r in range(rows):
            output[r][c] = color
    
    # Project scattered pixels matching divider colors
    for r in range(rows):
        for c in range(cols):
            pixel = grid[r][c]
            if pixel == bg or r in h_dividers or c in v_dividers:
                continue
            
            if pixel in divider_colors:
                # Find the closest divider of this color and project onto adjacent row/col
                # Check horizontal dividers
                for div_r, div_color in h_dividers.items():
                    if div_color == pixel:
                        # Project to adjacent row (toward the divider)
                        if r < div_r:
                            output[div_r - 1][c] = pixel
                        elif r > div_r:
                            output[div_r + 1][c] = pixel
                
                # Check vertical dividers
                for div_c, div_color in v_dividers.items():
                    if div_color == pixel:
                        # Project to adjacent column (toward the divider)
                        if c < div_c:
                            output[r][div_c - 1] = pixel
                        elif c > div_c:
                            output[r][div_c + 1] = pixel
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['60795d0c']
    
    all_pass = True
    for i, ex in enumerate(task['train']):
        result = transform(ex['input'])
        expected = ex['output']
        match = result == expected
        print(f"Train {i}: {'PASS' if match else 'FAIL'}")
        if not match:
            all_pass = False
            print("Expected:")
            for row in expected:
                print(row)
            print("Got:")
            for row in result:
                print(row)
    
    print(f"\nAll training examples: {'PASS' if all_pass else 'FAIL'}")
