#!/usr/bin/env python3
"""
Solver for ARC-AGI puzzle e99362f0

Pattern Analysis:
- Input is divided into 4 quadrants by a cross divider (row 5, column 4)
- Each quadrant has a distinct color: TL=7, TR=9, BL=2, BR=8
- Output is created by layering the masks with priority rules:

Priority logic (first match wins):
1. if BR=1: output=8
2. elif BL=1:
   - if TL=1: output=7
   - elif TR=1: output=9
   - else: output=2
3. elif TR=1:
   - if TL=1: output=7
   - else: output=9
4. elif TL=1: output=7
5. else: output=0
"""

def solve(grid):
    """
    Solve the puzzle by extracting quadrants and applying the layering logic.
    
    Args:
        grid: 11x9 input grid
    
    Returns:
        5x4 output grid
    """
    # Find dividers (row and column of all 4s)
    h_div = None
    v_div = None
    
    for i in range(len(grid)):
        if all(grid[i][j] == 4 for j in range(len(grid[0]))):
            h_div = i
            break
    
    for j in range(len(grid[0])):
        if all(grid[i][j] == 4 for i in range(len(grid))):
            v_div = j
            break
    
    if h_div is None or v_div is None:
        raise ValueError("Could not find dividers in grid")
    
    # Extract quadrants
    quad_tl = [row[:v_div] for row in grid[:h_div]]
    quad_tr = [row[v_div+1:] for row in grid[:h_div]]
    quad_bl = [row[:v_div] for row in grid[h_div+1:]]
    quad_br = [row[v_div+1:] for row in grid[h_div+1:]]
    
    # Create binary masks for each color
    tl_mask = [[1 if x == 7 else 0 for x in row] for row in quad_tl]
    tr_mask = [[1 if x == 9 else 0 for x in row] for row in quad_tr]
    bl_mask = [[1 if x == 2 else 0 for x in row] for row in quad_bl]
    br_mask = [[1 if x == 8 else 0 for x in row] for row in quad_br]
    
    # Apply layering logic
    output = []
    for i in range(len(quad_tl)):
        row = []
        for j in range(len(quad_tl[0])):
            tl = tl_mask[i][j]
            tr = tr_mask[i][j]
            bl = bl_mask[i][j]
            br = br_mask[i][j]
            
            if br == 1:
                val = 8
            elif bl == 1:
                if tl == 1:
                    val = 7
                elif tr == 1:
                    val = 9
                else:
                    val = 2
            elif tr == 1:
                if tl == 1:
                    val = 7
                else:
                    val = 9
            elif tl == 1:
                val = 7
            else:
                val = 0
            
            row.append(val)
        output.append(row)
    
    return output


if __name__ == "__main__":
    import json
    import sys
    
    # Test with the training examples
    with open('/Users/evanpieser/ARC_AMD_TRANSFER/data/ARC-AGI/data/evaluation/e99362f0.json') as f:
        task = json.load(f)
    
    all_pass = True
    for idx, ex in enumerate(task['train']):
        predicted = solve(ex['input'])
        expected = ex['output']
        
        match = predicted == expected
        status = "PASS" if match else "FAIL"
        print(f"Training example {idx}: {status}")
        
        if not match:
            all_pass = False
            print(f"  Expected: {expected}")
            print(f"  Got: {predicted}")
    
    if all_pass:
        print("\nAll training examples passed!")
        
        # Solve the test case
        test_input = task['test'][0]['input']
        test_output = solve(test_input)
        print(f"\nTest output:")
        for row in test_output:
            print(row)
    else:
        print("\nSome training examples failed!")
        sys.exit(1)
