def solve(grid):
    """
    ARC-AGI Task 76d965ef: Extract pattern and tile with fractal structure
    
    Rule: 
    - Output is 2x input dimensions
    - Extract non-background (non-3) pattern 
    - Tile in 3 horizontal segments: [pattern[0], pattern[0], pattern[row_idx]]
    """
    input_h, input_w = len(grid), len(grid[0])
    
    # Find non-background (non-3) pattern bounds
    min_r = max_r = min_c = max_c = None
    for r in range(input_h):
        for c in range(input_w):
            if grid[r][c] != 3:
                if min_r is None:
                    min_r = max_r = r
                    min_c = max_c = c
                else:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
    
    # Handle edge case: no pattern found
    if min_r is None:
        return [[3] * (input_w * 2) for _ in range(input_h * 2)]
    
    # Extract pattern
    pattern = []
    for r in range(min_r, max_r + 1):
        row = [grid[r][c] for c in range(min_c, max_c + 1)]
        pattern.append(row)
    
    pat_h, pat_w = len(pattern), len(pattern[0])
    out_h, out_w = input_h * 2, input_w * 2
    
    result = []
    
    for r in range(out_h):
        row = []
        
        # Current pattern row index (cycles through pattern)
        pattern_row_idx = r % pat_h
        
        # Segment 1: Always pattern[0]
        seg1 = pattern[0]
        
        # Segment 2: Always pattern[0]  
        seg2 = pattern[0]
        
        # Segment 3: Use corresponding pattern row
        seg3 = pattern[pattern_row_idx]
        
        # Build the complete row
        row = seg1 + seg2 + seg3
        
        # Truncate to exact output width if needed
        result.append(row[:out_w])
    
    return result


# Test function
def test_solve():
    import json
    
    # Load task data
    with open('/Users/evanpieser/apr12_tasks/76d965ef.json', 'r') as f:
        data = json.load(f)
    
    print("=== TESTING SOLVER ===")
    
    all_pass = True
    for i, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        # Run solver
        predicted_output = solve(input_grid)
        
        # Check if correct
        matches = predicted_output == expected_output
        
        print(f"\nTrain {i}:")
        print(f"  Input: {len(input_grid)}x{len(input_grid[0])}")
        print(f"  Expected: {len(expected_output)}x{len(expected_output[0])}")
        print(f"  Predicted: {len(predicted_output)}x{len(predicted_output[0])}")
        print(f"  PASS: {matches}")
        
        if not matches:
            all_pass = False
            print(f"  First few rows comparison:")
            for j in range(min(5, len(expected_output))):
                exp_row = expected_output[j]
                pred_row = predicted_output[j] if j < len(predicted_output) else "MISSING"
                match = exp_row == pred_row
                print(f"    Row {j}: Expected={exp_row[:10]}{'...' if len(exp_row) > 10 else ''}")
                print(f"           Predicted={pred_row[:10] if pred_row != 'MISSING' else 'MISSING'}{'...' if pred_row != 'MISSING' and len(pred_row) > 10 else ''}")
                print(f"           Match: {match}")
    
    print(f"\nOVERALL RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    test_solve()