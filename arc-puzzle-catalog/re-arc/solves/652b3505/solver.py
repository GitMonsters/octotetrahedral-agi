def transform(grid):
    """
    Pattern: The grid has colored markers on top and bottom rows (or left/right columns).
    For each column position, extend the top row's color downward and bottom row's color
    upward until they meet at a middle dividing row.
    """
    import copy
    grid = [list(row) for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Determine background color (most common)
    flat = [c for row in grid for c in row]
    bg = max(set(flat), key=flat.count)
    
    # Check if pattern is horizontal (top/bottom rows) or vertical (left/right columns)
    top_row = grid[0]
    bottom_row = grid[h-1]
    left_col = [grid[r][0] for r in range(h)]
    right_col = [grid[r][w-1] for r in range(h)]
    
    top_non_bg = sum(1 for c in top_row if c != bg)
    bottom_non_bg = sum(1 for c in bottom_row if c != bg)
    left_non_bg = sum(1 for c in left_col if c != bg)
    right_non_bg = sum(1 for c in right_col if c != bg)
    
    horiz_markers = top_non_bg + bottom_non_bg
    vert_markers = left_non_bg + right_non_bg
    
    output = copy.deepcopy(grid)
    
    if horiz_markers >= vert_markers:
        # Horizontal pattern: top and bottom rows define columns
        # Find the middle row (dividing line)
        mid = h // 2
        
        for col in range(w):
            top_val = grid[0][col]
            bottom_val = grid[h-1][col]
            
            # Fill from top down to mid
            for row in range(mid):
                output[row][col] = top_val
            
            # Fill from mid+1 to bottom
            for row in range(mid + 1, h):
                output[row][col] = bottom_val
            
            # Middle row: if both top and bottom are non-bg, use bg; otherwise keep pattern
            # Actually check the output - middle row seems to merge based on both having non-bg
            if top_val != bg and bottom_val != bg:
                output[mid][col] = bg
            elif top_val != bg:
                output[mid][col] = top_val
            elif bottom_val != bg:
                output[mid][col] = bottom_val
            else:
                output[mid][col] = bg
    else:
        # Vertical pattern: left and right columns define rows
        mid = w // 2
        
        for row in range(h):
            left_val = grid[row][0]
            right_val = grid[row][w-1]
            
            # Fill from left to mid
            for col in range(mid):
                output[row][col] = left_val
            
            # Fill from mid+1 to right
            for col in range(mid + 1, w):
                output[row][col] = right_val
            
            # Middle column
            if left_val != bg and right_val != bg:
                output[row][mid] = bg
            elif left_val != bg:
                output[row][mid] = left_val
            elif right_val != bg:
                output[row][mid] = right_val
            else:
                output[row][mid] = bg
    
    return [tuple(row) for row in output]


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['652b3505']
    
    print("Testing on training examples:")
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        # Convert to comparable format
        expected_tuples = [tuple(row) for row in expected]
        
        match = result == expected_tuples
        print(f"  Example {i}: {'PASS' if match else 'FAIL'}")
        
        if not match:
            all_pass = False
            print(f"    Expected: {expected[:2]}...")
            print(f"    Got:      {list(result[:2])}...")
    
    print(f"\nAll tests passed: {all_pass}")
