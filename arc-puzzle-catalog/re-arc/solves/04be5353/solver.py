"""
ARC Puzzle 04be5353 Solver

Pattern: 
1. Each non-background pixel extends to the right with alternating color,0,color,0...
2. When consecutive rows have same-color pixels:
   - If columns increase (rightward): shadow row above shows 0s at first row's color positions
   - If columns decrease (leftward): shadow row above shows 0 at rightmost color position before overlap
"""

import copy


def transform(grid):
    grid = [list(row) for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
    background = max(color_counts, key=color_counts.get)
    
    result = copy.deepcopy(grid)
    
    # Find pixels: row -> (col, color)
    pixels = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                pixels[r] = (c, grid[r][c])
    
    # Step 1: Extend each pixel's pattern to the right
    for r, (c, color) in pixels.items():
        for i, col in enumerate(range(c, cols)):
            if i % 2 == 0:
                result[r][col] = color
            else:
                result[r][col] = 0
    
    # Step 2: Find consecutive same-color pixel rows and add shadow
    pixel_rows = sorted(pixels.keys())
    i = 0
    while i < len(pixel_rows):
        r = pixel_rows[i]
        color = pixels[r][1]
        group = [r]
        j = i + 1
        while j < len(pixel_rows) and pixel_rows[j] == group[-1] + 1 and pixels[pixel_rows[j]][1] == color:
            group.append(pixel_rows[j])
            j += 1
        
        if len(group) >= 2:
            cols_list = [pixels[row][0] for row in group]
            
            # Check for monotonically increasing columns (rightward pattern)
            if all(cols_list[k] < cols_list[k+1] for k in range(len(cols_list)-1)):
                shadow_row = group[0] - 1
                if shadow_row >= 0 and shadow_row not in pixels:
                    first_col = cols_list[0]
                    # Add 0s at color positions (even distance from start)
                    for col in range(first_col, cols):
                        if (col - first_col) % 2 == 0:
                            result[shadow_row][col] = 0
            
            # Check for monotonically decreasing columns (leftward pattern)
            elif all(cols_list[k] > cols_list[k+1] for k in range(len(cols_list)-1)):
                shadow_row = group[0] - 1
                if shadow_row >= 0 and shadow_row not in pixels:
                    last_row_start = cols_list[-1]
                    first_row_start = cols_list[0]
                    # Find rightmost color position before the first row's start
                    rightmost = None
                    for col in range(last_row_start, first_row_start):
                        if (col - last_row_start) % 2 == 0:
                            rightmost = col
                    if rightmost is not None:
                        result[shadow_row][rightmost] = 0
        
        i = j if j > i else i + 1
    
    return result


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['04be5353']
    
    print("Testing on all training examples:\n")
    all_pass = True
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"Example {i}: {'✓ PASS' if match else '✗ FAIL'}")
        if not match:
            print(f"  Differences:")
            for r in range(len(expected)):
                for c in range(len(expected[0])):
                    if result[r][c] != expected[r][c]:
                        print(f"    Row {r}, Col {c}: got {result[r][c]}, expected {expected[r][c]}")
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
