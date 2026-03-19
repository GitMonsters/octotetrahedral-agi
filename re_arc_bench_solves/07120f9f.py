def transform(grid):
    import copy
    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)
    
    # Find background color (most common) and marker (different color in row 0)
    background = grid[0][0]
    marker_col = -1
    marker_color = -1
    
    for c in range(cols):
        if grid[0][c] != background:
            marker_col = c
            marker_color = grid[0][c]
            break
    
    if marker_col == -1:
        return result
    
    # Draw vertical stripes every 2 columns from marker_col to right edge
    stripe_cols = []
    for c in range(marker_col, cols, 2):
        stripe_cols.append(c)
        for r in range(rows):
            result[r][c] = marker_color
    
    # Every 4 columns from marker (0th, 2nd, 4th stripe...), add horizontal caps
    # at row 0 and row (rows-1), extending to the right by 2 cells
    for i, c in enumerate(stripe_cols):
        if i % 2 == 0:  # every other stripe (0, 2, 4...)
            # Top cap: fill positions c+1, c+2 if in bounds (row 0)
            # But only c+1 since c+2 is already a stripe
            # Actually looking at pattern: at these positions, also mark c+1
            if c + 1 < cols:
                result[0][c + 1] = marker_color
            if c + 2 < cols:
                result[0][c + 2] = marker_color
            # Bottom cap
            if c + 1 < cols:
                result[rows - 1][c + 1] = marker_color
            if c + 2 < cols:
                result[rows - 1][c + 2] = marker_color
    
    return result


if __name__ == "__main__":
    import json
    
    data = json.load(open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json'))
    puzzle = data['07120f9f']
    
    all_pass = True
    for i, example in enumerate(puzzle['train']):
        inp = example['input']
        expected = example['output']
        actual = transform(inp)
        
        if actual == expected:
            print(f"Train {i+1}: PASS")
        else:
            print(f"Train {i+1}: FAIL")
            print(f"Expected row 0: {expected[0]}")
            print(f"Actual row 0:   {actual[0]}")
            print(f"Expected last:  {expected[-1]}")
            print(f"Actual last:    {actual[-1]}")
            all_pass = False
    
    print(f"\nAll pass: {all_pass}")
