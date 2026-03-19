"""
ARC Puzzle 235b16b1 Solver

Pattern: Divide input grid into 3x3 regions. For each region, find any non-background
pixel and place it in the corresponding cell of a 3x3 output grid.
"""

def transform(grid):
    height = len(grid)
    width = len(grid[0])
    
    # Determine background color (most common value)
    from collections import Counter
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Initialize 3x3 output with background
    output = [[bg for _ in range(3)] for _ in range(3)]
    
    # For each non-background pixel, determine which 3x3 region it belongs to
    for r in range(height):
        for c in range(width):
            if grid[r][c] != bg:
                # Map to 3x3 output cell
                out_r = r * 3 // height
                out_c = c * 3 // width
                output[out_r][out_c] = grid[r][c]
    
    return output


if __name__ == "__main__":
    import json
    
    with open('/Users/evanpieser/Downloads/re-arc_test_challenges-2026-03-16T22-48-53.json') as f:
        data = json.load(f)
    
    task = data['235b16b1']
    
    print("Testing on ALL training examples:\n")
    all_pass = True
    
    for i, ex in enumerate(task['train']):
        inp = ex['input']
        expected = ex['output']
        result = transform(inp)
        
        match = result == expected
        all_pass = all_pass and match
        
        print(f"Train {i}: {'✓ PASS' if match else '✗ FAIL'}")
        if not match:
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
    
    print(f"\n{'All tests passed!' if all_pass else 'Some tests failed.'}")
